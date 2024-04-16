import copy
import logging
import os
from typing import List, Tuple

import dgl
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import Config
from utils.dataloader import MyDataLoader
from utils.entity import DataFeature, PairDataExample, SimpleDataFeature

logger = logging.getLogger(__name__)


def get_simple_features(dataloader: MyDataLoader, data_dir: str, key: str, max_seq_length: int, percentage: float,
                        tokenizer: PreTrainedTokenizer, overwrite_cache=False, cached=False) -> List[
    SimpleDataFeature]:
    if cached:
        cache_file_name = f"cached_{key}_simple_{tokenizer.__class__.__name__}_{max_seq_length}_{percentage}.pk"
        cache_file_path = os.path.join(data_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            logger.info(f"Loading cache file: {cache_file_path}")
            if not overwrite_cache:
                features = torch.load(cache_file_path)
                return features
            else:
                features = get_new_simple_features(dataloader, key, max_seq_length, tokenizer)
                logger.info(f"Overate the cached file: {cache_file_path}")
                torch.save(features, cache_file_path)
        else:
            logger.info(f"Cached file not exist, will create new one: {cache_file_path}")
            features = get_new_simple_features(dataloader, key, max_seq_length, tokenizer)
            torch.save(features, cache_file_path)
    else:
        features = get_new_simple_features(dataloader, key, max_seq_length, tokenizer)
    return features


def get_new_simple_features(dataloader: MyDataLoader, key: str, max_seq_length: int,
                            tokenizer: PreTrainedTokenizer) -> \
        List[SimpleDataFeature]:
    features = []
    total_count = 0
    truncated_count = 0

    t = tqdm(enumerate(dataloader), desc=f"Covert SimpleExamples to SimpleFeatures {key}", total=len(dataloader))
    for (index, example) in t:
        feature, s_total_count, s_truncated_count = covert_example_to_simple_feature(
            example, max_seq_length,
            tokenizer)
        total_count += s_total_count
        truncated_count += s_truncated_count

        t.set_description(
            f'convert examples to features, trun count: {truncated_count}, total_count: {total_count}, trun ratio: {truncated_count / total_count}')
        features.append(feature)
        t.update(1)
    return features


def covert_example_to_simple_feature(example: PairDataExample, max_seq_length, tokenizer) -> \
        Tuple[SimpleDataFeature, int, int]:
    _id = example.data_id
    input_ids, attention_masks, token_type_ids, s_total_count, s_truncated_count = encode_input_text(example,
                                                                                                     max_seq_length,
                                                                                                     tokenizer)
    feature = SimpleDataFeature(
        id=_id,
        input_ids=input_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        label=example.label
    )
    return feature, s_total_count, s_truncated_count


def get_features(dataloader: MyDataLoader, data_dir: str, key: str, max_seq_length: int, percentage: float,
                 tokenizer: PreTrainedTokenizer, overwrite_cache=False, cached=False) -> List[
    DataFeature]:
    features = []
    if cached:
        cache_file_name = f"cached_{key}_{tokenizer.__class__.__name__}_{max_seq_length}_{percentage}.pk"
        cache_file_path = os.path.join(data_dir, cache_file_name)
        if os.path.exists(cache_file_path):
            logger.info(f"Loading cache file: {cache_file_path}")
            if not overwrite_cache:
                features = torch.load(cache_file_path)
                return features
            else:
                features = get_new_features(dataloader, key, max_seq_length, tokenizer)
                logger.info(f"Overate the cached file: {cache_file_path}")
                torch.save(features, cache_file_path)
        else:
            logger.info(f"Cached file not exist, will create new one: {cache_file_path}")
            features = get_new_features(dataloader, key, max_seq_length, tokenizer)
            torch.save(features, cache_file_path)
    else:
        features = get_new_features(dataloader, key, max_seq_length, tokenizer)
    return features


def encode_input_text(example: PairDataExample, max_seq_length, tokenizer) \
        -> Tuple[List[int], List[int], List[int], int, int]:
    truncated_count = 0
    total_count = 0
    assert isinstance(example.example_1.sentence, str), f"Data Error: {example.example_1.sentence}"
    assert isinstance(example.example_2.sentence, str), f"Data Error: {example.example_2.sentence}"
    try:
        sentence_a = example.example_1.sentence
        sentence_b = example.example_2.sentence
        sentence = sentence_a + Config.SEP_TOKEN + sentence_b
        tok_results = tokenizer(sentence, add_special_tokens=True, max_length=max_seq_length,
                                padding="max_length", truncation="only_second", return_overflowing_tokens=True)
        if "num_truncated_tokens" in tok_results and tok_results["num_truncated_tokens"] > 0:
            logger.info(f"The seq_length is bigger than defined max_seq_length. Defined :{max_seq_length}"
                        f",truncated count:{tok_results['num_truncated_tokens']}")
            truncated_count += 1
        total_count += 1
        input_ids = tok_results["input_ids"]
        print(sentence)
        print(tokenizer.convert_ids_to_tokens(input_ids))
        assert len(input_ids) <= Config.data_args.max_seq_length, f"{len(input_ids)}--{input_ids}"

        sep_id = tokenizer.sep_token_id
        assert input_ids.count(sep_id) == 2, f"{sentence_a}-{sentence_b}-{tokenizer.convert_ids_to_tokens(input_ids)}"
        attention_masks = tok_results["attention_mask"]
        token_type_ids = tok_results["token_type_ids"] if "token_type_ids" in tok_results else None
        return input_ids, attention_masks, token_type_ids, total_count, truncated_count
    except Exception as e:
        print("Exception data:", example)
        print("Exception:", e)
        e.with_traceback()
        return [], [], [], 1, 0


def reorganise_graph_after_encode(amr_graph_truncated_count: int, example: PairDataExample, amr_input_ids, tokenizer):
    node_truncated_count = 0
    sep_id = tokenizer.convert_tokens_to_ids(Config.SEP_TOKEN)
    node_sep_id = tokenizer.convert_tokens_to_ids(Config.NODE_SEP_TOKEN)
    e1_input_ids, e1_nodes_intervals, e2_input_ids, e2_nodes_intervals = split_node_intervals(amr_input_ids, sep_id,
                                                                                              node_sep_id, example)
    nodes_intervals = copy.deepcopy(e1_nodes_intervals)
    nodes_intervals.extend(e2_nodes_intervals)
    if amr_graph_truncated_count == 0:
        nodes_nums = [len(example.example_1.nodes), len(example.example_2.nodes)]
        edges1 = example.example_1.edges
        edges2 = example.example_2.edges
    else:
        ori_node_ids_1 = [idx for idx in range(len(example.example_1.nodes))]
        cut_ed_node_ids_1 = [idx for idx in range(len(e1_nodes_intervals))]
        delete_node_ids_1 = [idx for idx in ori_node_ids_1 if idx not in cut_ed_node_ids_1]
        nodes1, edges2 = delete_graph_nodes_edges(example.example_1.nodes, example.example_1.edges, delete_node_ids_1)

        ori_node_ids_2 = [idx for idx in range(len(example.example_2.nodes))]
        cut_ed_node_ids_2 = [idx for idx in range(len(e2_nodes_intervals))]
        delete_node_ids_2 = [idx for idx in ori_node_ids_2 if idx not in cut_ed_node_ids_2]
        nodes2, edges1 = delete_graph_nodes_edges(example.example_2.nodes, example.example_2.edges, delete_node_ids_2)

        nodes_nums = [len(nodes1), len(nodes2)]
        node_truncated_count = len(delete_node_ids_1) + len(delete_node_ids_2)

    return nodes_nums, edges1, edges2, nodes_intervals, node_truncated_count


def delete_graph_nodes_edges(nodes, edges, delete_node_ids):
    new_nodes = [item for idx, item in enumerate(nodes) if idx not in delete_node_ids]
    src_nodes = [item[0] for item in edges]
    dst_nodes = [item[1] for item in edges]
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(nodes))
    graph.remove_nodes(delete_node_ids)
    assert graph.number_of_nodes() == len(new_nodes), \
        f"{graph.number_of_nodes()}--{len(new_nodes)}--{len(nodes)}--{delete_node_ids}"
    new_edges = list(zip(graph.edges()[0].numpy().tolist(), graph.edges()[1].numpy().tolist()))
    return nodes, new_edges


def split_node_intervals(nodes_input_ids, sep_id, node_sep_id, example):
    # 过滤掉tokenizer的max_seq_length补全
    nodes_input_ids = [item for item in nodes_input_ids if item != 1]
    split_q1_edu_id = []
    q1_amr_intervals = []
    split_q2_edu_id = []
    q2_amr_intervals = []
    amr_parts = split_by_value(nodes_input_ids, sep_id)
    for part_index, (part, start, end) in enumerate(amr_parts):
        # 根据node_sep_id再次分割每个部分
        node_parts = split_by_value(part, node_sep_id)
        for node_index, (node_part, node_start, node_end) in enumerate(node_parts):
            if part_index == 0:
                split_q1_edu_id.append(node_part)
                q1_amr_intervals.append([start + node_start, start + node_end])
            else:
                split_q2_edu_id.append(node_part)
                q2_amr_intervals.append([start + node_start, start + node_end])

    assert len(q1_amr_intervals) == len(example.example_1.nodes), \
        f"{len(q1_amr_intervals)}--{len(example.example_1.nodes)}--{q1_amr_intervals}--{example.example_1.nodes}--{nodes_input_ids}"

    assert len(q2_amr_intervals) == len(example.example_2.nodes), \
        f"{len(q2_amr_intervals)}--{len(example.example_2.nodes)}--{q2_amr_intervals}--{example.example_2.nodes}--{nodes_input_ids}"

    return split_q1_edu_id, q1_amr_intervals, split_q2_edu_id, q2_amr_intervals


def split_by_value(input_list, value):
    sublists = []
    current_sublist = []
    for i, item in enumerate(input_list):
        if item == value:
            if current_sublist:
                sublists.append((current_sublist, i - len(current_sublist), i - 1))
            current_sublist = []
        else:
            current_sublist.append(item)
    if current_sublist:
        sublists.append((current_sublist, len(input_list) - len(current_sublist), len(input_list) - 1))
    return sublists


def encode_input_amr_graph(example: PairDataExample, max_seq_length, tokenizer) -> Tuple[
    List[int], List[int], List[int], List[int], List[List[int]], List[List[int]], List[List[int]], int, int]:
    s_amr_truncated_count = 0

    try:
        if len(example.example_1.nodes) == 0:
            example.example_1.nodes.append(example.example_1.sentence)
        if len(example.example_2.nodes) == 0:
            example.example_2.nodes.append(example.example_2.sentence)

        sentence_a = Config.NODE_SEP_TOKEN.join(example.example_1.nodes)
        sentence_b = Config.NODE_SEP_TOKEN.join(example.example_2.nodes)

        tok_results = tokenizer(sentence_a, sentence_b, add_special_tokens=True, max_length=max_seq_length,
                                padding="max_length", truncation="only_second", return_overflowing_tokens=True)
        if "num_truncated_tokens" in tok_results and tok_results["num_truncated_tokens"] > 0:
            logger.info(f"The seq_length is bigger than defined max_seq_length. Defined :{max_seq_length}"
                        f",truncated count:{tok_results['num_truncated_tokens']}")
            s_amr_truncated_count += 1
        amr_input_ids = tok_results["input_ids"]

        assert len(amr_input_ids) <= Config.data_args.max_seq_length, f"{len(amr_input_ids)}--{amr_input_ids}"
        amr_attention_masks = tok_results["attention_mask"]
        amr_token_type_ids = tok_results["token_type_ids"] if "token_type_ids" in tok_results else None

        effective_tok = [item for item in amr_input_ids if item != 1]
        if len(effective_tok) > Config.max_tok_len:
            Config.max_tok_len = len(effective_tok)
            print("Max tok len:", Config.max_tok_len)

        nodes_nums, edges1, edges2, node_intervals, node_truncated_count = reorganise_graph_after_encode(
            s_amr_truncated_count, example,
            amr_input_ids, tokenizer)
        return amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums, edges1, edges2, node_intervals, s_amr_truncated_count, node_truncated_count
    except Exception as e:
        print("Exception data:", example)
        print("sentence_a:", sentence_a)
        print("sentence_b:", sentence_b)
        print("tok:", [tokenizer.convert_ids_to_tokens(amr_input_ids)])
        print("Exception:", e)
        e.with_traceback()
        return [], [], [], [0, 0], [], [], [], 0, 0
    pass


def fill_length(edges1, edges2, node_intervals):
    while len(edges1) < Config.max_edge_num:
        edges1.append([Config.node_intervals_padding_id, Config.node_intervals_padding_id])

    while len(edges2) < Config.max_edge_num:
        edges2.append([Config.node_intervals_padding_id, Config.node_intervals_padding_id])

    while len(node_intervals) < Config.max_node_num * 2:
        node_intervals.append([Config.node_intervals_padding_id, Config.node_intervals_padding_id])
    return edges1, edges2, node_intervals


def covert_example_to_feature(example: PairDataExample, max_seq_length, tokenizer) -> \
        Tuple[DataFeature, int, int, int, int]:
    _id = example.data_id
    input_ids, attention_masks, token_type_ids, s_total_count, s_truncated_count = encode_input_text(example,
                                                                                                     max_seq_length,
                                                                                                     tokenizer)
    amr_input_ids, amr_attention_masks, amr_token_type_ids, nodes_nums, edges1, edges2, node_intervals, s_amr_truncated_count, s_node_truncated_count = encode_input_amr_graph(
        example, max_seq_length, tokenizer)
    edges1, edges2, node_intervals = fill_length(edges1, edges2, node_intervals)
    feature = DataFeature(
        id=_id,
        input_ids=input_ids,
        attention_masks=attention_masks,
        token_type_ids=token_type_ids,
        amr_input_ids=amr_input_ids,
        amr_attention_masks=amr_attention_masks,
        amr_token_type_ids=amr_token_type_ids,
        nodes_nums=nodes_nums,
        edges1=edges1,
        edges2=edges2,
        node_intervals=node_intervals,
        label=example.label
    )
    return feature, s_total_count, s_truncated_count, s_amr_truncated_count, s_node_truncated_count


def get_new_features(dataloader: MyDataLoader, key: str, max_seq_length: int,
                     tokenizer: PreTrainedTokenizer) -> \
        List[DataFeature]:
    features = []
    total_count = 0
    truncated_count = 0
    node_truncated_count = 0
    amr_truncated_count = 0

    t = tqdm(enumerate(dataloader), desc=f"Covert SimpleExamples to SimpleFeatures {key}", total=len(dataloader))
    for (index, example) in t:
        feature, s_total_count, s_truncated_count, s_amr_truncated_count, s_node_truncated_count = covert_example_to_feature(
            example, max_seq_length,
            tokenizer)
        total_count += s_total_count
        truncated_count += s_truncated_count
        node_truncated_count += s_node_truncated_count
        amr_truncated_count += s_amr_truncated_count

        t.set_description(
            f'convert examples to features, trun count: {truncated_count}, trun amr count:{amr_truncated_count},  trun_node_count:{node_truncated_count}, total_count: {total_count}, trun ratio: {truncated_count / total_count}')
        features.append(feature)
        t.update(1)
    return features
