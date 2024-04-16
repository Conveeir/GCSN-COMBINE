import concurrent
import csv
import json
import os.path
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import dgl
import hanlp
import pandas as pd
import requests
from tqdm import tqdm


#
def test_tok():
    tokenize_word_model_path = "F:/models/coarse_electra_small"

    tok = hanlp.load(tokenize_word_model_path)
    sentence = "How do I overcome my pornography addiction?"
    tok_sentence = tok(sentence)
    print(tok_sentence)


# def test_amr():
#     # test_data = test_tok()
#     sentence = "What would happen if Bill Gates bought three billion dollars worth of stock and then sold them all at once?"
#     tok_sentence = tok(sentence)
#     print(tok_sentence)
#     amr = amr_parser(tok_sentence, output_amr=False, language="eng")
#     print(len(amr))
#     print(amr)


def get_tok_sentence(sentences):
    tokenize_word_model_path = "F:/models/coarse_electra_small"

    tok = hanlp.load(tokenize_word_model_path)
    error_ids = []
    try:
        tok_sentences = tok(sentences)
    except Exception as e:
        print(e)
        tok_sentences = []
        for idx, sentence in enumerate(sentences):
            try:
                tok_sentences.append(tok(sentence))
            except Exception as e:
                print("Error Sentence:", idx, sentence)
                print(e)
                error_ids.append(idx)
                tok_sentences.append([""])
    # print(res)
    return tok_sentences, error_ids


def convert_amr_to_graph_example(amrs):
    graph_datas = []
    for amr in amrs:
        graph_data = {"nodes": [], "adj": [], "nodes_num": 0}
        nodes = []
        adj = []
        for amr_node in amr["nodes"]:
            nodes.append(amr_node["label"])
        for edge in amr["edges"]:
            adj.append((edge["source"], edge["target"]))
        nodes_num = len(nodes)
        graph_data["nodes"] = nodes
        graph_data["adj"] = adj
        graph_data["nodes_num"] = nodes_num
        graph_datas.append(graph_data)
    return graph_datas


def get_example_data_from_csv(data_csv_file_name="../datasets/cleaned/question.csv"):
    csv_datas = pd.read_csv(data_csv_file_name)
    datas = []
    for csv_data in csv_datas.values:
        line_data = csv_data.tolist()
        data_1 = {"id": line_data[0], "question_id": line_data[1], "question": line_data[3], "label": line_data[5]}
        data_2 = {"id": line_data[0], "question_id": line_data[2], "question": line_data[4], "label": line_data[5]}
        datas.append(data_1)
        datas.append(data_2)
    return datas


def convert_data_tok_data(datas):
    batch_size = 10
    total_step = int(len(datas) / batch_size)
    final_batch_len = len(datas) - (batch_size * total_step)

    total_tasks = total_step + (1 if final_batch_len > 0 else 0)
    try:

        for index in range(total_step):
            batch_datas = datas[index * batch_size:(index + 1) * batch_size]

            sentences = [item["question"] for item in batch_datas]
            # 在这里对每个批次的数据进行处理
            tok_sentences, error_idx = get_tok_sentence(sentences)
            for j in range(0, len(batch_datas)):
                if j in error_idx:
                    datas[index * batch_size + j]["tok"] = [""]
                else:
                    datas[index * batch_size + j]["tok"] = tok_sentences[j]

        if final_batch_len > 0:
            last_batch_datas = datas[(total_step - 1) * batch_size:]

            sentences = [item["question"] for item in last_batch_datas]
            # 在这里对每个批次的数据进行处理
            tok_sentences, error_idx = get_tok_sentence(sentences)
            for j in range(0, len(last_batch_datas)):
                if j in error_idx:
                    datas[index * batch_size + j]["tok"] = [""]
                else:
                    datas[index * batch_size + j]["tok"] = tok_sentences[j]

    except Exception as e:
        print(e)
    finally:
        save_file_path = "../datasets/tok/no_tok.json"
        save_json_file(datas, save_file_path)


def process_data(data, amr_parser):
    try:
        data["amr"] = amr_parser(data["tok"], output_amr=False, language="eng")
    except Exception as e:
        print(data)
        print(e)
    return data


def convert_data_amr_data(datas, save_file_path):
    amr_model_path = "F:/models/amr-eng-zho-xlm-roberta-base_20220412_223756"
    amr_parser = hanlp.load(amr_model_path, devices=0)

    for index in tqdm(range(len(datas))):
        datas[index] = process_data(datas[index], amr_parser)
    save_json_file(datas, save_file_path)


def save_json_file(datas, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(datas, f)


def load_json_file(path, file_name):
    file_path = os.path.join(path, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        datas = json.load(f)
    return datas


# 针对漏处理的数据用小数据集进行处理，随后合并
def merge_data(full_datas, sub_datas, key, path, file_name):
    merged_count = 0
    sub_data_dic = {}
    for item in sub_datas:
        sub_data_dic[item["question_id"]] = item
    for idx, data in enumerate(full_datas):
        if key not in data.keys():
            data_qid = data["question_id"]
            if key in sub_data_dic[data_qid].keys():
                full_datas[idx] = sub_data_dic[data_qid]
                merged_count += 1
    print("Merged Count:", merged_count)
    file_path = os.path.join(path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(full_datas, f)


# 检测处理完毕后的数据是否完整
def check_preprocess_data(path, file_name):
    file_path = os.path.join(path, file_name)
    with open("file_path", "r", encoding="utf-8") as f:
        datas = json.load(f)
    get_no_tok_datas(datas)
    get_no_amr_datas(datas)


def get_no_tok_datas(datas):
    no_tok_datas = []
    for data in tqdm(datas):
        if "tok" not in data.keys():
            no_tok_datas.append(data)
    print(len(no_tok_datas))
    with open("datasets/amr/no_tok_data.json", "w", encoding="utf-8") as f:
        json.dump(no_tok_datas, f)


def get_no_amr_datas(datas):
    no_amr_datas = []
    for data in tqdm(datas):
        if "amr" not in data.keys():
            no_amr_datas.append(data)
    print(len(no_amr_datas))
    with open("datasets/amr/no_amr_data.json", "w", encoding="utf-8") as f:
        json.dump(no_amr_datas, f)


def clean_ori_csv_data():
    datas = get_example_data_from_csv()
    datas = filter_datas(datas)
    save_new_file(datas)


# 清洗原始csv数据，删除空白数据
def get_example_data_from_csv():
    data_csv_file_name = "../ori_data/cleaned/question.csv"
    csv_datas = pd.read_csv(data_csv_file_name)
    datas = []
    for csv_data in csv_datas.values:
        line_data = csv_data.tolist()
        data = {"id": line_data[0], "qid1": line_data[1], "qid2": line_data[2], "question1": line_data[3],
                "question2": line_data[4], "is_duplicate": line_data[5]}
        datas.append(data)
    return datas


def save_new_file(datas):
    df = pd.DataFrame(datas)
    df.to_csv("cleaned/question.csv", index=False)


def empty_question(value):
    return value is None or value == "" or isinstance(value, float) or len(value) == 0


def filter_datas(datas):
    error_data_ids = []
    for index, data in enumerate(datas):
        if empty_question(data["question1"]) or empty_question(data["question2"]):
            error_data_ids.append(index)
    datas = [item for index, item in enumerate(datas) if index not in error_data_ids]
    print("Delete datas:", len(error_data_ids), error_data_ids)
    return datas


def split_datas(path, all_file_name, train_file_name, test_file_name, dev_file_name, test_rate, dev_rate):
    all_file = os.path.join(path, all_file_name)
    train_file = os.path.join(path, train_file_name)
    test_file = os.path.join(path, test_file_name)
    dev_file = os.path.join(path, dev_file_name)

    with open(all_file, "r", encoding="utf-8") as f:
        all_datas = json.load(f)

    # 随机打乱数据
    random.shuffle(all_datas)

    # 计算数据集划分的大小
    total_size = len(all_datas)
    dev_size = int(dev_rate * total_size)
    test_size = int(test_rate * total_size)
    train_size = total_size - dev_size - test_size

    # 划分数据集
    train_dataset = all_datas[:train_size]

    test_dataset = all_datas[train_size:train_size + test_size]

    dev_dataset = all_datas[train_size + test_size:]

    # 确保划分后的数据集大小总和等于原始数据集大小
    assert len(train_dataset) + len(test_dataset) + len(dev_dataset) == total_size

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_dataset, f)

    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f)

    with open(dev_file, "w", encoding="utf-8") as f:
        json.dump(dev_dataset, f)


def convert_pair_data(all_datas, path, file_name):
    pair_datas = []
    for idx in tqdm(range(0, int(len(all_datas) / 2))):
        pair_datas.append([all_datas[2 * idx], all_datas[2 * idx + 1]])
    save_json_file(pair_datas, os.path.join(path, file_name))


def convert_data_to_example(all_datas, path, file_name):
    tar_datas = []
    for data in tqdm(all_datas):
        data_1 = data[0]
        nodes_1, edges_1 = get_nodes_edges_from_amr(data_1["amr"])
        tar_data_1 = {"id": data_1["id"], "question": data_1["question"], "nodes": nodes_1, "adj": edges_1}

        data_2 = data[1]
        nodes_2, edges_2 = get_nodes_edges_from_amr(data_2["amr"])
        tar_data_2 = {"id": data_2["id"], "question": data_2["question"], "nodes": nodes_2, "adj": edges_2}

        tar_datas.append([tar_data_1, tar_data_2])
    with open(os.path.join(path, file_name), "w", encoding="utf-8") as f:
        json.dump(tar_datas, f)


def get_nodes_edges_from_amr(amr: dict):
    """
        amr_json = {"id": "0", "input": "What is the step by step guide to invest in share market in india ?",
                "nodes": [{"id": 0, "label": "step-by-step"}, {"id": 1, "label": "guide-01"},
                          {"id": 2, "label": "invest-01"}, {"id": 3, "label": "share"}, {"id": 4, "label": "market"},
                          {"id": 5, "label": "country"}, {"id": 6, "label": "amr-unknown"},
                          {"id": 7, "label": "name", "properties": ["op1"], "values": ["india"]}],
                "edges": [{"source": 5, "target": 7, "label": "name"}, {"source": 1, "target": 6, "label": "arg0"},
                          {"source": 3, "target": 4, "label": "domain"}, {"source": 1, "target": 2, "label": "arg2"},
                          {"source": 2, "target": 4, "label": "arg2"}, {"source": 4, "target": 5, "label": "location"},
                          {"source": 0, "target": 1, "label": "domain"}, {"source": 1, "target": 0, "label": "manner"}],
                "tops": [1], "framework": "amr"}
    :param amr:
    :return:
    """
    nodes = []
    for item in amr["nodes"]:
        nodes.append(item["label"])
    edges = []
    for item in amr["edges"]:
        edges.append([item["source"], item["target"]])
    return nodes, edges


def clean_data_nodes(datas, path, file_name):
    total_tasks = len(datas)
    for idx in range(total_tasks):
        nodes, edges = delete_empty_nodes(datas[idx][0]["nodes"], datas[idx][0]["adj"])
        datas[idx][0]["nodes"] = nodes
        datas[idx][0]["adj"] = edges

        nodes, edges = delete_empty_nodes(datas[idx][1]["nodes"], datas[idx][1]["adj"])
        datas[idx][1]["nodes"] = nodes
        datas[idx][1]["adj"] = edges

    print(f"Finished {os.path.join(path, file_name)}")
    with open(os.path.join(path, file_name), "w", encoding="utf-8") as f:
        json.dump(datas, f)


def delete_empty_nodes(nodes, edges):
    delete_node_ids = []
    for idx, node in enumerate(nodes):
        if len(node) == 0 or "amr-unknown" in node:
            delete_node_ids.append(idx)

    src_nodes = [item[0] for item in edges]
    dst_nodes = [item[1] for item in edges]

    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=len(nodes))
    graph.remove_nodes(delete_node_ids)

    new_edges = list(zip(graph.edges()[0].numpy().tolist(), graph.edges()[1].numpy().tolist()))
    new_nodes = [item for idx, item in enumerate(nodes) if idx not in delete_node_ids]
    return new_nodes, new_edges


def filter_nodes(datas, path, file_name):
    node_node_threshold = 20
    deleted_data_id = []
    for data in datas:
        if len(data[0]["nodes"]) > node_node_threshold or len(data[1]["nodes"]) > node_node_threshold:
            deleted_data_id.append(data[0]["id"])
    print("Deleted data: ", len(deleted_data_id), len(deleted_data_id) / len(datas), deleted_data_id)
    new_datas = [item for item in datas if item[0]["id"] not in deleted_data_id]
    save_json_file(new_datas, os.path.join(path, file_name))


def update_label(data, csv_data_dict):
    data_id = data[0]["id"]
    if data_id in csv_data_dict:
        data[0]["label"] = csv_data_dict[data_id]


def fill_label(datas, csv_datas, path, file_name):
    csv_data_dict = {data["id"]: data["is_duplicate"] for data in csv_datas}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(update_label, data, csv_data_dict) for data in datas]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Updating Labels"):
            pass

    with open(os.path.join(path, file_name), "w", encoding="utf-8") as f:
        json.dump(datas, f)
    print(f"Finished {os.path.join(path, file_name)}")


def load_and_clean_data(key):
    datas = load_json_file("../datasets", f"{key}.json")

    cleaned_datas = []
    for idx in tqdm(range(len(datas))):
        nodes, edges = delete_empty_nodes(datas[idx][0]["nodes"], datas[idx][0]["adj"])
        datas[idx][0]["nodes"] = nodes
        datas[idx][0]["adj"] = edges

        nodes, edges = delete_empty_nodes(datas[idx][1]["nodes"], datas[idx][1]["adj"])
        datas[idx][1]["nodes"] = nodes
        datas[idx][1]["adj"] = edges

    cleaned_datas.append(datas)
    return cleaned_datas


if __name__ == '__main__':

    import multiprocessing as mp
    from functools import partial

    pool = mp.Pool()
    keys = ["train", "test", "dev"]
    params = [(key,) for key in keys]
    # 使用starmap方法，这样可以解包元组并传递给load_and_clean_data
    cleaned_data = pool.starmap(load_and_clean_data, params)
    pool.close()
    pool.join()

    for key, data in zip(keys, cleaned_data):
        save_json_file(data, os.path.join("../datasets/cleaned", f"{key}.json"))
    # csv_datas = get_example_data_from_csv()
    # datas = load_json_file("../datasets/amr", "no_tok_data.json")
    # convert_data_tok_data(datas)
    # save_json_file(datas, save_file_path)

    # with open("../datasets/tok/all_tok.json", "r", encoding="utf-8") as f:
    #     datas = json.load(f)
    #     save_file_path = "../datasets/amr/no_amr.json"
    #     convert_data_amr_data(datas)

    # with open("../datasets/tok/no_tok.json", "r", encoding="utf-8") as f:
    #     datas = json.load(f)
    #     convert_data_amr_data(datas)
    # test_tok()
    # full_datas = load_json_file("../datasets/amr", "merge_tok.json")
    # sub_datas = load_json_file("../datasets/amr", "no_amr.json")
    # merge_data(full_datas, sub_datas, "amr", "../datasets/amr", "merge_amr.json")
    # all_datas = load_json_file("../datasets/amr", "all.json")
    # convert_pair_data(all_datas, "../datasets/pair_data", "all.json")
    # convert_data_to_example(all_datas, "../datasets", "all.json")
    # all_datas = load_json_file("../datasets", "all.json")
    # filter_nodes(all_datas, "../datasets/short", "all.json")
    # split_datas("../datasets/short", "all.json", "train.json", "test.json", "dev.json", 0.1, 0.1)
    # train_datas = load_json_file("../ori_data/cleaned_nodes", "train.json")
    # # clean_data_nodes(train_datas, "../datasets/cleaned_nodes", "train.json")
    # fill_label(train_datas, csv_datas, "../ori_data/label", "train.json")
    #
    # dev_datas = load_json_file("../ori_data/cleaned_nodes", "dev.json")
    # # clean_data_nodes(dev_datas, "../datasets/cleaned_nodes", "dev.json")
    # fill_label(dev_datas, csv_datas, "../ori_data/label", "dev.json")
    #
    # test_datas = load_json_file("../ori_data/cleaned_nodes", "test.json")
    # # clean_data_nodes(test_datas, "../datasets/cleaned_nodes", "test.json")
    # fill_label(test_datas, csv_datas, "../ori_data/label", "test.json")
