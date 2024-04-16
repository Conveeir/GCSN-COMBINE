import copy
import json

import dgl
import networkx as nx
import torch
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from torch import nn


def load_example_data():
    with open("example.json", "r", encoding="utf-8") as f:
        datas = json.load(f)
    graph_1_nodes_num = datas[0]["nodes_num"]
    graph_1_edges = datas[0]["edges"]
    src_1 = [item[0] for item in graph_1_edges]
    dst_1 = [item[1] for item in graph_1_edges]
    graph_1_nodes_vectors = torch.tensor(datas[0]["nodes_vector"])
    graph1 = dgl.graph((src_1, dst_1), num_nodes=graph_1_nodes_num)
    graph1 = dgl.to_bidirected(graph1)
    graph1.ndata['h'] = graph_1_nodes_vectors
    nx_graph1 = graph1.to_networkx().to_undirected()
    assert nx.is_connected(nx_graph1)

    graph_2_nodes_num = datas[1]["nodes_num"]
    graph_2_edges = datas[1]["edges"]
    src_2 = [item[0] for item in graph_2_edges]
    dst_2 = [item[1] for item in graph_2_edges]
    graph_2_nodes_vectors = torch.tensor(datas[1]["nodes_vector"])
    graph2 = dgl.graph((src_2, dst_2), num_nodes=graph_2_nodes_num)
    graph2 = dgl.to_bidirected(graph2)
    graph2.ndata['h'] = graph_2_nodes_vectors
    nx_graph2 = graph2.to_networkx().to_undirected()
    assert nx.is_connected(nx_graph2)

    print(graph1)
    print(graph1.ndata["h"])
    print(graph2)
    print(graph2.ndata["h"])
    return graph1, graph2


def get_feature_by_gcn(graph):
    hidden_size = 1024
    nodes_features = graph.ndata['h']
    print("nodes_features:", nodes_features)
    adj_matrix = get_adj_matrix_by_graph(graph)
    degree_matrix = get_degree_matrix(graph)
    adj_matrix_ba = norm_laplace_transform(adj_matrix, degree_matrix, graph.number_of_nodes())

    conv1 = dglnn.GraphConv(hidden_size, hidden_size)
    w_1 = torch.randn(hidden_size, hidden_size)

    conv2 = dglnn.GraphConv(hidden_size, hidden_size)
    w_2 = torch.randn(hidden_size, hidden_size)

    # =========================
    nodes_features = conv1(graph, nodes_features)
    adj_ba_x = torch.mm(adj_matrix_ba, nodes_features)
    adj_ba_x_w = torch.mm(adj_ba_x, w_1)
    layer1_out = torch.relu(adj_ba_x_w)
    # =========================

    print("layer1_out:", layer1_out)

    # =========================
    nodes_features = conv2(graph, layer1_out)
    adj_ba_x = torch.mm(adj_matrix_ba, nodes_features)
    adj_ba_x_w = torch.mm(adj_ba_x, w_2)
    layer2_out = torch.relu(adj_ba_x_w)
    # =========================
    print("layer2_out:", layer2_out)

    graph.ndata['h'] = layer2_out
    return graph


def get_adj_matrix_by_graph(graph: dgl.DGLGraph):
    adj_matrix = graph.adjacency_matrix(scipy_fmt="coo")

    # 将邻接矩阵转换为 PyTorch Tensor
    adj_matrix = torch.Tensor(adj_matrix.toarray())

    return adj_matrix


def get_degree_matrix(graph: dgl.DGLGraph):
    return torch.diag(graph.in_degrees().float())


def norm_laplace_transform(adj_matrix, degree_matrix, num_nodes):
    sqrt_inv_deg = torch.diag(torch.sqrt(torch.diag(1.0 / torch.sqrt(degree_matrix))))
    sym_norm_laplacian_matrix = torch.eye(num_nodes) - torch.mm(torch.mm(sqrt_inv_deg, adj_matrix),
                                                                sqrt_inv_deg)
    return sym_norm_laplacian_matrix


def node_graph_matching(graph1, graph2):
    hidden_size = 1024
    graph_1_nodes_num = graph1.number_of_nodes()
    graph_1_nodes_hidden_states = graph1.ndata["h"]

    graph_2_nodes_num = graph2.number_of_nodes()
    graph_2_nodes_hidden_states = graph2.ndata["h"]

    alpha = get_alpha(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states)
    beta = get_beta(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states)
    print("alpha:", alpha.size(), alpha)
    print("beta:", beta.size(), beta)

    h_g_2_avg = compute_avg_graph_to_each_nodes_1(alpha, graph_1_nodes_num, graph_2_nodes_num,
                                                  graph_2_nodes_hidden_states)

    print("h_g_2_avg:", h_g_2_avg.size(), h_g_2_avg)

    h_g_1_avg = compute_avg_graph_to_each_nodes_2(beta, graph_1_nodes_num, graph_2_nodes_num,
                                                  graph_1_nodes_hidden_states)
    print("h_g_1_avg:", h_g_1_avg.size(), h_g_1_avg)

    w_m = torch.randn(hidden_size, hidden_size)

    graph_1_node_h_interacted_graph_2 = get_nodes_graph_interaction(graph_1_nodes_num, graph_1_nodes_hidden_states,
                                                                    h_g_2_avg, w_m)
    print("graph_1_node_h_interacted_graph_2:", graph_1_node_h_interacted_graph_2.size(),
          graph_1_node_h_interacted_graph_2)
    graph_2_node_h_interacted_graph_1 = get_nodes_graph_interaction(graph_2_nodes_num, graph_2_nodes_hidden_states,
                                                                    h_g_1_avg, w_m)
    print("graph_2_node_h_interacted_graph_1:", graph_2_node_h_interacted_graph_1.size(),
          graph_2_node_h_interacted_graph_1)
    graph1.ndata["h"] = graph_1_node_h_interacted_graph_2

    graph2.ndata["h"] = graph_2_node_h_interacted_graph_1
    return graph1, graph2


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


def get_graph_hidden_states(graph1, graph2):
    hidden_size = 1024
    bilstm = nn.LSTM(hidden_size, hidden_size, 1, bidirectional=True, batch_first=True)

    output1, _ = bilstm(graph1.ndata["h"].unsqueeze(0))
    # 取双向LSTM的输出，连接两个方向的最后两个隐藏向量
    forward_output_1 = output1[:, -1, :hidden_size]
    backward_output_1 = output1[:, 0, hidden_size:]
    h_g_tilde_1 = torch.cat((forward_output_1, backward_output_1), dim=1)
    h_g_tilde_1 = h_g_tilde_1.squeeze(0)

    output2, _ = bilstm(graph2.ndata["h"].unsqueeze(0))
    # 取双向LSTM的输出，连接两个方向的最后两个隐藏向量
    forward_output_2 = output2[:, -1, :hidden_size]
    backward_output_2 = output2[:, 0, hidden_size:]
    h_g_tilde_2 = torch.cat((forward_output_2, backward_output_2), dim=1)
    h_g_tilde_2 = h_g_tilde_2.squeeze(0)

    return h_g_tilde_1, h_g_tilde_2


def prediction(h_g_tilde_1, h_g_tilde_2):
    return cosine_similarity(h_g_tilde_1, h_g_tilde_2)


if __name__ == '__main__':
    graph1, graph2 = load_example_data()
    graph1 = get_feature_by_gcn(graph1)
    graph2 = get_feature_by_gcn(graph2)
    graph1, graph2 = node_graph_matching(graph1, graph2)
    h_g_tilde_1, h_g_tilde_2 = get_graph_hidden_states(graph1, graph2)
    y_tilde = prediction(h_g_tilde_1, h_g_tilde_2)
    print(y_tilde.size(), y_tilde)
