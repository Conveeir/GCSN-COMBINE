from transformers import RobertaTokenizer

import Config
from utils.dataloader import MyDataLoader
from utils.dataset import MyDataset


def test_data_loader():
    data_dir = "datasets"

    tokenizer_class = RobertaTokenizer
    # 加载RoBERTa模型的 tokenizer  从预训练模型地址中加载
    tokenizer = tokenizer_class.from_pretrained(
        r"F:\models\roberta-large",
        cache_dir="cache",
    )
    tokenizer.add_tokens("-")
    tokenizer.add_special_tokens({"additional_special_tokens": [Config.SEP_TOKEN, Config.NODE_SEP_TOKEN]})

    train_data_loader = MyDataLoader(data_dir, "train", 1.0, False, True)
    print(train_data_loader[0])
    train_dataset = MyDataset(train_data_loader, data_dir, "dev", Config.max_seq_length, tokenizer, True, True)
    print(train_dataset[0])

    dev_data_loader = MyDataLoader(data_dir, "dev", 1.0, False, True)
    print(dev_data_loader[0])
    dev_dataset = MyDataset(dev_data_loader, data_dir, "dev", Config.max_seq_length, tokenizer, True, True)
    print(dev_dataset[0])

    test_data_loader = MyDataLoader(data_dir, "test", 1.0, False, True)
    print(test_data_loader[0])
    test_dataset = MyDataset(test_data_loader, data_dir, "test", Config.max_seq_length, tokenizer, True, True)
    print(test_dataset[0])


if __name__ == '__main__':
    test_data_loader()
