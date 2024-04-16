import json
import logging
import os

import dgl
import networkx as nx
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import Config
from utils.entity import SingleDataExample, PairDataExample

logger = logging.getLogger(__name__)


def get_examples(data_dir: str, key: str, percentage: float, overwrite_cache=True, cached=False):
    if cached:
        cache_file_name = f"cached_{key}_{percentage}.pk"
        cache_file_path = os.path.join(data_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            logger.info(f"Loading cache file: {cache_file_path}")
            if not overwrite_cache:
                examples = torch.load(cache_file_path)
                return examples
            else:
                examples = get_new_examples(data_dir, key, percentage)
                logger.info(f"Overate the cached file: {cache_file_path}")
                torch.save(examples, cache_file_path)
        else:
            logger.info(f"Cached file not exist, will create new one: {cache_file_path}")
            examples = get_new_examples(data_dir, key, percentage)
            torch.save(examples, cache_file_path)
    else:
        examples = get_new_examples(data_dir, key, percentage)
    return examples


def get_new_examples(data_dir: str, key: str, percentage: float):
    datas = load_json_data(data_dir, key, percentage)
    examples = []
    max_node_num = 0
    max_edge_num = 0
    node_threshold = 20
    bigger_than_th_ids = []

    max_node_data_id = -1
    max_edge_data_id = -1
    for index, data in tqdm(enumerate(datas), total=len(datas), desc=f"Create TLG Examples {key}"):
        example = create_example(data, index, data_dir)
        if len(example.example_1.nodes) > node_threshold:
            bigger_than_th_ids.append((example.data_id, len(example.example_1.nodes)))
        if len(example.example_2.nodes) > node_threshold:
            bigger_than_th_ids.append((example.data_id, len(example.example_2.nodes)))

        if len(example.example_1.nodes) > max_node_num:
            max_node_num = len(example.example_1.nodes)
            max_node_data_id = example.data_id

        if len(example.example_2.nodes) > max_node_num:
            max_node_num = len(example.example_2.nodes)
            max_node_data_id = example.data_id

        if len(example.example_1.edges) > max_edge_num:
            max_edge_num = len(example.example_1.edges)
            max_edge_data_id = example.data_id

        if len(example.example_2.edges) > max_edge_num:
            max_edge_num = len(example.example_2.edges)
            max_edge_data_id = example.data_id

        examples.append(example)

    if max_edge_num > Config.max_edge_num:
        Config.max_edge_num = max_edge_num
    if max_node_num > Config.max_node_num:
        Config.max_node_num = max_node_num
    print(f"Max node_num: {max_node_num}\t Max edge_num: {max_edge_num}")
    return examples


def create_example(data: list, index: int, data_dir: str) -> PairDataExample:
    # print(data)
    nodes_1 = data[0]["nodes"]
    edges_1 = data[0]["adj"]
    sentence_1 = data[0]["question"]
    label = data[0]["label"]

    example_1 = SingleDataExample(
        nodes=nodes_1,
        edges=edges_1,
        sentence=sentence_1
    )

    nodes_2 = data[1]["nodes"]
    edges_2 = data[1]["adj"]
    sentence_2 = data[1]["question"]

    example_2 = SingleDataExample(
        nodes=nodes_2,
        edges=edges_2,
        sentence=sentence_2
    )

    example = PairDataExample(
        data_id=data[0]["id"],
        example_1=example_1,
        example_2=example_2,
        label=label,
    )
    return example


def get_percentage_datas(datas, percentage: float):
    if percentage <= 0 or percentage > 1:
        raise ValueError("Percentage should be in the range (0, 1]")

    num_elements = int(len(datas) * percentage)
    if num_elements < 1:
        num_elements = 1  # Ensure at least one element is selected

    selected_elements = datas[:num_elements]
    return selected_elements


def load_json_data(data_dir: str, key: str, percentage: float):
    data_file_path = os.path.join(data_dir, f"{key}.json")
    logger.info(f"Loading data_file: {data_file_path}")
    with open(data_file_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    datas = get_percentage_datas(datas, percentage)
    return datas


def is_dgl_graph_connected(graph, show_graph_or_not=False):
    nx_graph = dgl.to_networkx(graph).to_undirected()
    ret = nx.is_connected(nx_graph)
    if not ret and show_graph_or_not:
        show_graph(nx_graph)
    return ret


def show_graph(nx_graph):
    nx.draw_networkx(nx_graph)
    plt.title("Graph")
    plt.show()
