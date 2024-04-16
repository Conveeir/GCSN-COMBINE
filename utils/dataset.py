from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils.dataloader import MyDataLoader
from utils.feature_process import get_features, get_simple_features


class MyDataset(Dataset):

    def __init__(self, dataloader: MyDataLoader, data_dir: str, key: str, max_seq_length: int,
                 tokenizer: PreTrainedTokenizer, percentage: float, overwrite_cache=False, cached=False):
        self.dataloader = dataloader
        self.features = get_features(dataloader=self.dataloader, data_dir=data_dir, key=key,
                                     max_seq_length=max_seq_length, tokenizer=tokenizer,
                                     overwrite_cache=overwrite_cache, cached=cached, percentage=percentage)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class MySimpleDataset(Dataset):
    def __init__(self, dataloader: MyDataLoader, data_dir: str, key: str, max_seq_length: int,
                 tokenizer: PreTrainedTokenizer, percentage: float, overwrite_cache=False, cached=False):
        self.dataloader = dataloader
        self.features = get_simple_features(dataloader=self.dataloader, data_dir=data_dir, key=key,
                                            max_seq_length=max_seq_length, tokenizer=tokenizer,
                                            overwrite_cache=overwrite_cache, cached=cached, percentage=percentage)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
