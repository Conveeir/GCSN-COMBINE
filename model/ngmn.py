import time

import dgl
import networkx as nx
import torch
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch import nn
import torch.multiprocessing as mp

from model.gcn_layer import MyGCNLayer


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间：{execution_time} 秒")
        return result

    return wrapper


class NGMN(nn.Module):
    def __init__(self, hidden_size, gnn_num_layer, dropout):
        super(NGMN, self).__init__()
        self.gcn_layers = nn.ModuleList()
        self.gnn_num_layer = gnn_num_layer
        self.hidden_size = hidden_size
        for _ in range(gnn_num_layer):
            self.gcn_layers.append(MyGCNLayer(hidden_size, dropout))
        self.bilstm = nn.LSTM(hidden_size, hidden_size, 1, bidirectional=True, batch_first=True)
        self.w_m = torch.randn(hidden_size, hidden_size)


    def forward(self, batch_graphs1, batch_graphs2):
        device = self.get_data_device(batch_graphs1)
        batch_graphs1 = self.get_feature_by_gcn(batch_graphs1, device)
        batch_graphs2 = self.get_feature_by_gcn(batch_graphs2, device)

        batch_graphs1, batch_graphs2 = self.node_graph_matching(batch_graphs1, batch_graphs2, device)

        h_g_tilde_1, h_g_tilde_2 = self.get_graph_hidden_states(batch_graphs1, batch_graphs2, device)

        model_outputs = self.prediction(h_g_tilde_1, h_g_tilde_2)
        return model_outputs


    def get_data_device(self, batch_graphs):
        graphs_list = dgl.unbatch(batch_graphs)
        device = f"cuda:{graphs_list[0].ndata['h'].get_device()}"
        return device


    def prediction(self, h_g_tilde_1, h_g_tilde_2):
        print(h_g_tilde_1.size())
        print(h_g_tilde_2.size())
        outputs = F.cosine_similarity(h_g_tilde_1, h_g_tilde_2, dim=1)
        return outputs


    def get_feature_by_gcn(self, batch_graphs, device):
        graphs_list = dgl.unbatch(batch_graphs)
        assert self.hidden_size == graphs_list[0].ndata["h"].size(-1)
        g_nodes = torch.Tensor([item.number_of_nodes() for item in graphs_list]).int()
        adj_matrix_list = get_adj_matrix_by_graph_batch(batch_graphs, g_nodes, device)
        degree_matrix_list = get_degree_matrix_graph_batch(batch_graphs, g_nodes)
        norm_laplace_transform_list = [norm_laplace_transform(a_i, d_i, n_i, device) for idx, (a_i, d_i, n_i) in
                                       enumerate(zip(adj_matrix_list, degree_matrix_list, g_nodes))]
        for i in range(len(graphs_list)):
            adj_matrix_ba = norm_laplace_transform_list[i]
            for l_i in range(self.gnn_num_layer):
                graphs_list[i] = self.gcn_layers[l_i](graphs_list[i], adj_matrix_ba, device)
        batch_graphs = dgl.batch(graphs_list)
        return batch_graphs


    def node_graph_matching(self, batch_graph1, batch_graph2, device):

        self.w_m = self.w_m.to(device)
        graph1_list = dgl.unbatch(batch_graph1)
        graph2_list = dgl.unbatch(batch_graph2)
        assert len(graph1_list) == len(graph2_list)
        num_graphs = len(graph1_list)
        for idx in range(num_graphs):
            graph1 = graph1_list[idx]
            graph2 = graph2_list[idx]
            assert graph1.ndata['h'].size(-1) == graph2.ndata['h'].size(-1)

            graph_1_nodes_num = graph1.number_of_nodes()
            graph_1_nodes_hidden_states = graph1.ndata["h"]

            graph_2_nodes_num = graph2.number_of_nodes()
            graph_2_nodes_hidden_states = graph2.ndata["h"]

            alpha = get_alpha(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states,
                              graph_2_nodes_hidden_states)


            beta = get_beta(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states,
                            graph_2_nodes_hidden_states)


            h_g_2_avg = compute_avg_graph_to_each_nodes_1(alpha, graph_1_nodes_num, graph_2_nodes_num,
                                                          graph_2_nodes_hidden_states)

            h_g_1_avg = compute_avg_graph_to_each_nodes_2(beta, graph_1_nodes_num, graph_2_nodes_num,
                                                          graph_1_nodes_hidden_states)

            graph_1_node_h_interacted_graph_2 = get_nodes_graph_interaction(graph_1_nodes_num,
                                                                            graph_1_nodes_hidden_states,
                                                                            h_g_2_avg, self.w_m)

            graph_2_node_h_interacted_graph_1 = get_nodes_graph_interaction(graph_2_nodes_num,
                                                                            graph_2_nodes_hidden_states,
                                                                            h_g_1_avg, self.w_m)

            graph1.ndata["h"] = graph_1_node_h_interacted_graph_2
            graph1_list[idx] = graph1

            graph2.ndata["h"] = graph_2_node_h_interacted_graph_1
            graph2_list[idx] = graph2
        batch_graph1 = dgl.batch(graph1_list)
        batch_graph2 = dgl.batch(graph2_list)

        return batch_graph1, batch_graph2

    def get_graph_hidden_states(self, batch_graph1, batch_graph2, device):
        hidden_size = self.hidden_size
        graph_1_list = dgl.unbatch(batch_graph1)
        graph_2_list = dgl.unbatch(batch_graph2)
        self.bilstm.to(device)
        h_g_tilde_1s = []
        h_g_tilde_2s = []
        assert len(graph_1_list) == len(graph_2_list)
        batch_size = len(graph_1_list)
        for idx in range(len(graph_1_list)):
            batch_size_num = 1

            input_data_1 = graph_1_list[idx].ndata["h"].view(1, -1, hidden_size)
            output1, _ = self.bilstm(input_data_1)
            # 取双向LSTM的输出，连接两个方向的最后两个隐藏向量
            forward_output_1 = output1[:, -1, :hidden_size]
            backward_output_1 = output1[:, 0, hidden_size:]
            h_g_tilde_1 = torch.cat((forward_output_1, backward_output_1), dim=1)
            h_g_tilde_1s.append(h_g_tilde_1)

            input_data_2 = graph_2_list[idx].ndata["h"].view(1, -1, hidden_size)
            output2, _ = self.bilstm(input_data_2)
            # 取双向LSTM的输出，连接两个方向的最后两个隐藏向量
            forward_output_2 = output2[:, -1, :hidden_size]
            backward_output_2 = output2[:, 0, hidden_size:]
            h_g_tilde_2 = torch.cat((forward_output_2, backward_output_2), dim=1)
            h_g_tilde_2s.append(h_g_tilde_2)

        return torch.stack(h_g_tilde_1s, dim=1).view(batch_size, -1), torch.stack(h_g_tilde_2s, dim=1).view(batch_size,
                                                                                                            -1)


def get_adj_matrix_by_graph_batch(batch_graphs, g_nodes, device):
    adj_matrix = batch_graphs.adjacency_matrix(scipy_fmt="coo")

    # 将邻接矩阵转换为 PyTorch Tensor
    adjs = torch.Tensor(adj_matrix.toarray()).to(device)

    adj_matrices = []
    start_idx = 0
    for num_nodes in g_nodes:
        end_idx = start_idx + num_nodes
        adj_matrix = adjs[start_idx:end_idx, start_idx:end_idx]
        adj_matrices.append(adj_matrix)
        start_idx = end_idx
    return adj_matrices


def get_degree_matrix_graph_batch(batch_graphs, g_nodes):
    res_degrees = []
    degrees = torch.diag(batch_graphs.in_degrees().float())
    start_idx = 0
    for num_nodes in g_nodes:
        end_idx = start_idx + num_nodes
        degree = degrees[start_idx:end_idx, start_idx:end_idx]
        res_degrees.append(degree)
        start_idx = end_idx
    return res_degrees


def norm_laplace_transform(adj_matrix, degree_matrix, num_nodes, device):
    sqrt_inv_deg = torch.diag(torch.sqrt(torch.diag(1.0 / torch.sqrt(degree_matrix))))
    sym_norm_laplacian_matrix = torch.eye(num_nodes).to(device) - torch.mm(torch.mm(sqrt_inv_deg, adj_matrix),
                                                                           sqrt_inv_deg)
    return sym_norm_laplacian_matrix


def get_alpha(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states):
    alpha = torch.zeros((graph_1_nodes_num, graph_2_nodes_num))

    for i in range(graph_1_nodes_num):
        for j in range(graph_2_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            alpha[i][j] = cosine_similarity(h_i_1, h_j_2)
    return alpha


def get_beta(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states):
    beta = torch.zeros((graph_2_nodes_num, graph_1_nodes_num))

    for j in range(graph_2_nodes_num):
        for i in range(graph_1_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            beta[j][i] = cosine_similarity(h_j_2, h_i_1)
    return beta


def cosine_similarity(h1, h2):
    # 计算余弦相似度
    similarity = F.cosine_similarity(h1, h2, dim=0)
    return similarity


def compute_avg_graph_to_each_nodes_1(alpha, graph_1_nodes_num, graph_2_nodes_num, graph_2_nodes_hidden_states):
    h_g_2_avg = []
    for i in range(graph_1_nodes_num):
        h_g_2_i_s = []
        for j in range(graph_2_nodes_num):
            alph_i_j = alpha[i][j]
            h_2_j = graph_2_nodes_hidden_states[j]
            h_g_2_i_j = alph_i_j * h_2_j
            h_g_2_i_s.append(h_g_2_i_j)
        h_g_2_i_s = torch.stack(h_g_2_i_s, dim=0)
        h_g_2_i_avg = torch.sum(h_g_2_i_s, dim=0)
        h_g_2_avg.append(h_g_2_i_avg)
    return torch.stack(h_g_2_avg)


def compute_avg_graph_to_each_nodes_2(beta, graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states):
    h_g_1_avg = []
    for j in range(graph_2_nodes_num):
        h_g_1_j_s = []
        for i in range(graph_1_nodes_num):
            beta_j_i = beta[j][i]
            h_1_j = graph_1_nodes_hidden_states[i]
            h_g_1_j_i = beta_j_i * h_1_j
            h_g_1_j_s.append(h_g_1_j_i)
        h_g_1_j_s = torch.stack(h_g_1_j_s, dim=0)
        h_g_2_i_avg = torch.sum(h_g_1_j_s, dim=0)
        h_g_1_avg.append(h_g_2_i_avg)
    return torch.stack(h_g_1_avg)


def get_nodes_graph_interaction(nodes_num, nodes_hidden_states, h_g_l_avg, w_m):
    nodes_graph_interaction = []
    for idx in range(nodes_num):
        h_idx_l = nodes_hidden_states[idx]
        h_g_l_idx_avg = h_g_l_avg[idx]
        nodes_graph_interaction.append(f_m(h_idx_l, h_g_l_idx_avg, w_m))
    return torch.stack(nodes_graph_interaction)


def f_m(x1, x2, wk):
    # 计算两个向量的 cosine 相似度
    cos_sim = F.cosine_similarity(x1 * wk, x2 * wk, dim=1)
    return cos_sim
