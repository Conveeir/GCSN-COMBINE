import time

import dgl
import networkx as nx
import numpy as np
import torch
from torch import nn
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import MultipleChoiceModelOutput, SequenceClassifierOutput

import Config
from model.ngmn import NGMN


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间：{execution_time} 秒")
        return result

    return wrapper


class MyMatchModel(RobertaPreTrainedModel):
    def __init__(self, lm_config, max_seq_length: int, dropout: float, gnn_layer_num: int, label_nums):
        super().__init__(lm_config)
        self.max_seq_length = max_seq_length
        self.label_nums = label_nums
        self.dropout = nn.Dropout(dropout)
        self.roberta = RobertaModel(self.config)
        # self.output_layer = nn.Linear(lm_config.hidden_size * 2, 2)
        self.output_layer = nn.Linear(lm_config.hidden_size, 2)
        self.ngmn = NGMN(lm_config.hidden_size, gnn_layer_num, dropout)
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.CrossEntropyLoss()
        pass

    def forward(self,
                id,
                input_ids,
                labels,
                attention_masks,
                amr_input_ids,
                amr_attention_masks,
                nodes_nums,
                edges1,
                edges2,
                node_intervals,
                token_type_ids=None,
                amr_token_type_ids=None
                ):
        # input_ids, attention_masks, last_hidden_states = self.get_text_representations(input_ids, attention_masks,
        #                                                                                token_type_ids)
        print(input_ids)
        print(amr_input_ids)
        device = f'cuda:{input_ids.get_device()}'
        amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums, amr_last_hidden_states = \
            self.get_edu_text_representations(amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums)
        batch_graphs_1, graphs_1_nodes_num, batch_graphs_2, graphs_2_nodes_num = self.get_graphs_from_batch_data(
            nodes_nums, edges1, edges2, device)
        #
        # question_1_representation, question_2_representation = self.get_split_question_representations(input_ids,
        #                                                                                                last_hidden_states)
        # print(last_hidden_states.size())
        # print(last_hidden_states[:0].size())
        #
        batch_graphs_1 = self.init_graphs(amr_last_hidden_states, node_intervals, batch_graphs_1, graphs_1_nodes_num)
        batch_graphs_2 = self.init_graphs(amr_last_hidden_states, node_intervals, batch_graphs_2, graphs_2_nodes_num)

        mgnn_output = self.ngmn(batch_graphs_1, batch_graphs_2)
        #
        # input_for_linear = torch.cat([question_1_representation, question_2_representation],
        #                              dim=-1)
        # input_for_linear = last_hidden_states[:, 0, :]
        #
        # outputs = self.output_layer(input_for_linear).view(-1, self.label_nums)

        loss = self.loss_fn(mgnn_output, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=mgnn_output,
        )

    def get_split_question_representations(self, input_ids, last_hidden_states):
        sep_id = torch.tensor(Config.tokenizer.convert_tokens_to_ids(Config.SEP_TOKEN))
        # [CLS] question1 [SEP] question2 [SEP]

        sep_locs = [np.where((input_id == sep_id).view(-1).detach().cpu().numpy())[0].tolist() for input_id in
                    input_ids]

        sep_interval = [[(1, sep_locs[index][0]), (sep_locs[index][0] + 1, sep_locs[index][1])] for index in
                        range(len(sep_locs))]

        question_1_representations = []
        question_2_representations = []
        for index, si in enumerate(sep_interval):
            question_1_representations.append(
                torch.mean(last_hidden_states[index, si[0][0]:si[0][1], :], dim=0).view(self.config.hidden_size))
            question_2_representations.append(
                torch.mean(last_hidden_states[index, si[1][0]:si[1][1], :], dim=0).view(self.config.hidden_size))

        return torch.stack(question_1_representations), torch.stack(question_2_representations)

    def get_text_representations(self, input_ids, attention_masks, token_type_ids=None):
        # input_ids = input_ids.view(-1, input_ids.size(-1))
        # attention_masks = attention_masks.view(-1, attention_masks.size(-1))
        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        robert_outputs = self.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden_states = robert_outputs['last_hidden_state']
        return input_ids, attention_masks, last_hidden_states

    def get_edu_text_representations(self, amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums):
        amr_input_ids = amr_input_ids.view(-1, amr_input_ids.size(-1))
        amr_attention_masks = amr_attention_masks.view(-1, amr_attention_masks.size(-1))
        if amr_token_type_ids is not None:
            amr_token_type_ids = amr_token_type_ids.view(-1, amr_token_type_ids.size(-1))
        nodes_nums = nodes_nums.view(-1, nodes_nums.size(-1))

        amr_robert_outputs = self.roberta(
            amr_input_ids,
            attention_mask=amr_attention_masks,
            token_type_ids=amr_token_type_ids,
            return_dict=True
        )
        amr_last_hidden_states = amr_robert_outputs['last_hidden_state']
        return amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums, amr_last_hidden_states

    def get_graphs_from_batch_data(self, nodes_nums, edges1, edges2, device):
        edges1 = edges1.detach().cpu().numpy().tolist()
        edges2 = edges2.detach().cpu().numpy().tolist()

        batch_graphs_1 = []
        graphs_1_nodes_num = []
        batch_graphs_2 = []
        graphs_2_nodes_num = []
        for data_idx in range(len(nodes_nums)):
            graph_1_nodes = nodes_nums[data_idx][0]
            graph_1_edges = [item for item in edges1[data_idx] if item[0] != Config.node_intervals_padding_id]
            graph_1_edges_dgl_type_src_nodes = [item[0] for item in graph_1_edges]
            graph_1_edges_dgl_type_dst_nodes = [item[1] for item in graph_1_edges]
            graph_1 = dgl.graph((graph_1_edges_dgl_type_src_nodes, graph_1_edges_dgl_type_dst_nodes),
                                num_nodes=graph_1_nodes)

            graph_1 = dgl.to_bidirected(graph_1)
            graph_1 = dgl.add_self_loop(graph_1)

            batch_graphs_1.append(graph_1)
            graphs_1_nodes_num.append(graph_1_nodes)

            graph_2_nodes = nodes_nums[data_idx][1]
            graph_2_edges = [item for item in edges2[data_idx] if item[0] != Config.node_intervals_padding_id]
            graph_2_edges_dgl_type_src_nodes = [item[0] for item in graph_2_edges]
            graph_2_edges_dgl_type_dst_nodes = [item[1] for item in graph_2_edges]
            graph_2 = dgl.graph((graph_2_edges_dgl_type_src_nodes, graph_2_edges_dgl_type_dst_nodes),
                                num_nodes=graph_2_nodes)
            graph_2 = dgl.to_bidirected(graph_2)
            graph_2 = dgl.add_self_loop(graph_2)

            batch_graphs_2.append(graph_2)
            graphs_2_nodes_num.append(graph_2_nodes)

        batch_graphs_1 = dgl.batch(batch_graphs_1).to(device)
        batch_graphs_2 = dgl.batch(batch_graphs_2).to(device)
        return batch_graphs_1, graphs_1_nodes_num, batch_graphs_2, graphs_2_nodes_num

    def init_graphs(self, amr_last_hidden_states, node_intervals, graphs, graph_nodes):
        nodes_representations = self.init_nodes(amr_last_hidden_states, node_intervals,
                                                graph_nodes)
        graphs.ndata["h"] = torch.cat(nodes_representations, dim=0)
        return graphs

    def init_nodes(self, amr_last_hidden_states, node_intervals, node_nums):
        """
        node_intervals: batch_size counts data
                        [
                            [ [data1.x1,data1.y1],[data1.x2,data1.y2] ],  all nodes of i-th data
                            [ [data2.x1,data2.y1],[data2.x2,data2.y2] ],
                        ]
        amr_last_hidden_states: batch_size counts data
                        [
                            [ [0.111, ..., 0.11 (hidden_size counts item)], ..., [] ] max_seq_length counts
                        ]

        """
        node_intervals = node_intervals.detach().cpu().numpy().tolist()
        node_intervals = [node_interval[0:node_nums[index]] for index, node_interval in
                          enumerate(node_intervals)]
        amr_node_representations = []
        for d_idx in range(len(node_intervals)):
            e_data_node_representations_list = []
            for (s, e) in node_intervals[d_idx]:
                # 获取每个节点所属区间内的 嵌入的平均值
                node_representation = torch.mean(amr_last_hidden_states[d_idx, s:e + 1, :], dim=0).view(
                    self.config.hidden_size)
                # 确保没有nan的出现
                assert check_nan_in_tensor(node_representation), \
                    f"{(s, e)}"
                e_data_node_representations_list.append(node_representation)
            # 得到单个图中所有节点的初始化嵌入值
            e_data_node_representations = torch.stack(e_data_node_representations_list)
            assert len(e_data_node_representations) == node_nums[
                d_idx], f"{len(e_data_node_representations)}--{node_nums[d_idx]}"
            # 得到这批数据的所有图的初始化嵌入值
            amr_node_representations.append(e_data_node_representations)

        return amr_node_representations


def check_nan_in_tensor(data):
    if torch.isnan(data).any():
        print(data.detach().cpu().numpy().tolist())

    return not torch.isnan(data).any()
