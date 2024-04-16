from utils.data_process import get_examples
from utils.entity import PairDataExample


class MyDataLoader:
    def __init__(self, data_dir: str, key: str, percentage: float, overwrite_cache, cached):
        self.examples = get_examples(data_dir, key, percentage, overwrite_cache, cached)
        self.index = 0

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> PairDataExample:
        return self.examples[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.examples):
            item = self.examples[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration
