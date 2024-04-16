import time
import torch.multiprocessing as mp
import dgl

print(dgl.__version__)
import torch
import dgl.function as fn
from dgl.udf import EdgeBatch, NodeBatch
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间：{execution_time} 秒")
        return result

    return wrapper


@timing_decorator
def node_graph_matching():
    nodes_h = [[1], [2], [3], [4]]
    edges = [[0, 1], [0, 2], [0, 3]]
    graph_1 = dgl.graph(edges, num_nodes=len(nodes_h))
    graph_1 = dgl.to_bidirected(graph_1)
    graph_1.ndata['h'] = torch.Tensor(nodes_h)

    nodes_h = [[1], [2], [3], [4], [5]]
    edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
    graph_2 = dgl.graph(edges, num_nodes=len(nodes_h))
    graph_2 = dgl.to_bidirected(graph_2)
    graph_2.ndata['h'] = torch.Tensor(nodes_h)

    batch_graphs1 = dgl.batch(
        [graph_1, graph_1, graph_1, graph_2, graph_1, graph_1, graph_2, graph_1, graph_2, graph_2])
    batch_graphs2 = dgl.batch(
        [graph_1, graph_2, graph_1, graph_1, graph_2, graph_2, graph_1, graph_2, graph_2, graph_2])
    graph1_list = dgl.unbatch(batch_graphs1)
    graph2_list = dgl.unbatch(batch_graphs2)

    assert len(graph1_list) == len(graph2_list)
    num_graphs = len(graph1_list)
    for idx in range(num_graphs):
        graph1 = graph1_list[idx]
        graph2 = graph2_list[idx]
        assert graph1.ndata['h'].size(-1) == graph2.ndata['h'].size(-1)
        hidden_size = graph1.ndata['h'].size(-1)

        graph_1_nodes_num = graph1.number_of_nodes()
        graph_1_nodes_hidden_states = graph1.ndata["h"]

        graph_2_nodes_num = graph2.number_of_nodes()
        graph_2_nodes_hidden_states = graph2.ndata["h"]

        alpha = get_alpha(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states,
                          graph_2_nodes_hidden_states)
        beta = get_beta(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states)

        h_g_2_avg = compute_avg_graph_to_each_nodes_1(alpha, graph_1_nodes_num, graph_2_nodes_num,
                                                      graph_2_nodes_hidden_states)

        h_g_1_avg = compute_avg_graph_to_each_nodes_2(beta, graph_1_nodes_num, graph_2_nodes_num,
                                                      graph_1_nodes_hidden_states)

        w_m = torch.randn(hidden_size, hidden_size)

        graph_1_node_h_interacted_graph_2 = get_nodes_graph_interaction(graph_1_nodes_num, graph_1_nodes_hidden_states,
                                                                        h_g_2_avg, w_m)

        graph_2_node_h_interacted_graph_1 = get_nodes_graph_interaction(graph_2_nodes_num, graph_2_nodes_hidden_states,
                                                                        h_g_1_avg, w_m)

        graph1.ndata["h"] = graph_1_node_h_interacted_graph_2
        graph1_list[idx] = graph1

        graph2.ndata["h"] = graph_2_node_h_interacted_graph_1
        graph2_list[idx] = graph2
    batch_graph1 = dgl.batch(graph1_list)
    batch_graph2 = dgl.batch(graph2_list)
    return batch_graph1, batch_graph2


def get_alpha(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states):
    alpha = torch.zeros((graph_1_nodes_num, graph_2_nodes_num))

    for i in range(graph_1_nodes_num):
        for j in range(graph_2_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            alpha[i][j] = F.cosine_similarity(h_i_1, h_j_2, dim=0)
    return alpha


def get_beta(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states):
    beta = torch.zeros((graph_2_nodes_num, graph_1_nodes_num))

    for j in range(graph_2_nodes_num):
        for i in range(graph_1_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            beta[j][i] = F.cosine_similarity(h_j_2, h_i_1, dim=0)
    return beta


def get_alpha_pool(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states,
                   result_queue):
    alpha = torch.zeros((graph_1_nodes_num, graph_2_nodes_num))

    for i in range(graph_1_nodes_num):
        for j in range(graph_2_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            alpha[i][j] = F.cosine_similarity(h_i_1, h_j_2, dim=0)
    result_queue.put(alpha)


def get_beta_pool(graph_1_nodes_num, graph_2_nodes_num, graph_1_nodes_hidden_states, graph_2_nodes_hidden_states,
                  result_queue):
    beta = torch.zeros((graph_2_nodes_num, graph_1_nodes_num))

    for j in range(graph_2_nodes_num):
        for i in range(graph_1_nodes_num):
            h_i_1 = graph_1_nodes_hidden_states[i]
            h_j_2 = graph_2_nodes_hidden_states[j]
            beta[j][i] = F.cosine_similarity(h_j_2, h_i_1, dim=0)
    result_queue.put(beta)


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


@timing_decorator
def node_graph_matching_pool():
    nodes_h = [[1], [2], [3], [4]]
    edges = [[0, 1], [0, 2], [0, 3]]
    graph_1 = dgl.graph(edges, num_nodes=len(nodes_h))
    graph_1 = dgl.to_bidirected(graph_1)
    graph_1.ndata['h'] = torch.Tensor(nodes_h)

    nodes_h = [[1], [2], [3], [4], [5]]
    edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
    graph_2 = dgl.graph(edges, num_nodes=len(nodes_h))
    graph_2 = dgl.to_bidirected(graph_2)
    graph_2.ndata['h'] = torch.Tensor(nodes_h)

    batch_graphs1 = dgl.batch(
        [graph_1, graph_1, graph_1, graph_2, graph_1, graph_1, graph_2, graph_1, graph_2, graph_2])
    batch_graphs2 = dgl.batch(
        [graph_1, graph_2, graph_1, graph_1, graph_2, graph_2, graph_1, graph_2, graph_2, graph_2])
    graph1_list = dgl.unbatch(batch_graphs1)
    graph2_list = dgl.unbatch(batch_graphs2)
    assert len(graph1_list) == len(graph2_list)

    num_graphs = len(graph1_list)
    processes = []
    result_queue = mp.Queue()

    for idx in range(num_graphs):
        process = mp.Process(target=get_alpha_pool, args=(
            graph1_list[idx].number_of_nodes(), graph2_list[idx].number_of_nodes(),
            graph1_list[idx].ndata['h'], graph2_list[idx].ndata['h'], result_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    alphas = [result_queue.get() for _ in processes]

    # 清空结果队列
    while not result_queue.empty():
        result_queue.get()

    processes = []  # 清空进程列表

    for idx in range(num_graphs):
        process = mp.Process(target=get_beta_pool, args=(
            graph1_list[idx].number_of_nodes(), graph2_list[idx].number_of_nodes(),
            graph1_list[idx].ndata['h'], graph2_list[idx].ndata['h'], result_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    betas = [result_queue.get() for _ in processes]

    for idx in range(num_graphs):
        graph1 = graph1_list[idx]
        graph2 = graph2_list[idx]
        assert graph1.ndata['h'].size(-1) == graph2.ndata['h'].size(-1)
        hidden_size = graph1.ndata['h'].size(-1)

        graph_1_nodes_num = graph1.number_of_nodes()
        graph_1_nodes_hidden_states = graph1.ndata["h"]

        graph_2_nodes_num = graph2.number_of_nodes()
        graph_2_nodes_hidden_states = graph2.ndata["h"]

        alpha = alphas[idx]
        beta = betas[idx]

        h_g_2_avg = compute_avg_graph_to_each_nodes_1(alpha, graph_1_nodes_num, graph_2_nodes_num,
                                                      graph_2_nodes_hidden_states)

        h_g_1_avg = compute_avg_graph_to_each_nodes_2(beta, graph_1_nodes_num, graph_2_nodes_num,
                                                      graph_1_nodes_hidden_states)

        w_m = torch.randn(hidden_size, hidden_size)

        graph_1_node_h_interacted_graph_2 = get_nodes_graph_interaction(graph_1_nodes_num, graph_1_nodes_hidden_states,
                                                                        h_g_2_avg, w_m)

        graph_2_node_h_interacted_graph_1 = get_nodes_graph_interaction(graph_2_nodes_num, graph_2_nodes_hidden_states,
                                                                        h_g_1_avg, w_m)

        graph1.ndata["h"] = graph_1_node_h_interacted_graph_2
        graph1_list[idx] = graph1

        graph2.ndata["h"] = graph_2_node_h_interacted_graph_1
        graph2_list[idx] = graph2
    batch_graph1 = dgl.batch(graph1_list)
    batch_graph2 = dgl.batch(graph2_list)
    print(batch_graph1, batch_graph2)


if __name__ == '__main__':
    mp.set_start_method("spawn")
    node_graph_matching_pool()
    # print(node_graph_matching(batch_graphs1, batch_graphs2))
