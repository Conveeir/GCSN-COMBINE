import json

from tqdm import tqdm


def get_no_tok_datas():
    no_tok_datas = []
    with open("datasets/amr/merge_amr.json", "r", encoding="utf-8") as f:
        datas = json.load(f)
        for data in tqdm(datas):
            if "tok" not in data.keys():
                no_tok_datas.append(data)
    print(len(no_tok_datas))
    with open("datasets/amr/no_tok_data.json", "w", encoding="utf-8") as f:
        json.dump(no_tok_datas, f)


def get_no_amr_datas():
    no_amr_datas = []
    with open("datasets/amr/merge_amr.json", "r", encoding="utf-8") as f:
        datas = json.load(f)
        for data in tqdm(datas):
            if "amr" not in data.keys():
                no_amr_datas.append(data)
    print(len(no_amr_datas))
    with open("datasets/amr/no_amr_data.json", "w", encoding="utf-8") as f:
        json.dump(no_amr_datas, f)


if __name__ == '__main__':
    get_no_tok_datas()
    get_no_amr_datas()
