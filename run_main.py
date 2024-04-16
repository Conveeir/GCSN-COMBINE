import logging
import os
from dataclasses import dataclass, field

import dgl
import numpy as np
import torch.cuda
from transformers import TrainingArguments, Trainer, HfArgumentParser, AutoConfig, \
    RobertaTokenizer
from transformers.trainer_utils import set_seed, EvalPrediction, PredictionOutput

import Config
from datasets import load_metric
from model.graph_match import MyMatchModel
from utils.dataloader import MyDataLoader
from utils.dataset import MyDataset, MySimpleDataset

logger = logging.getLogger(__name__)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"torch version:{torch.__version__}")
print(f"dgl version: {dgl.__version__}")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="F:/models/roberta-large",
                                    metadata={"help": "RoBERTa or BERT models' path or name"})
    dropout: float = field(default=0.1, metadata={"help": "Dropout rate"})
    gnn_layers_num: int = field(default=2)
    do_test: bool = field(default=False)
    check_points: str = field(default="output/checkpoint-38572/")
    result_dir: str = field(default="results")
    do_eval_first: bool = field(default=False)


@dataclass
class DataArguments:
    data_dir: str = field(default="../datasets")
    max_seq_length: int = field(default=160)
    data_percentage: float = field(
        default=0.01,
        metadata={"help": "percentage of datas. 1.0 for all datas"}
    )
    data_cached: bool = field(
        default=True,
        metadata={"help": "Use or not use cached files"}
    )
    data_overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    if (
            os.path.exists(train_args.output_dir)
            and os.listdir(train_args.output_dir)
            and train_args.do_train
            and not train_args.overwrite_output_dir
    ): raise ValueError(
        f"Output directory ({train_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if train_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        train_args.local_rank,
        train_args.device,
        train_args.n_gpu,
        bool(train_args.local_rank != -1),
        train_args.fp16,
    )
    set_seed(train_args.seed)
    return model_args, data_args, train_args


def set_config(model_args, data_args, train_args, tokenizer):
    Config.model_args = model_args
    Config.data_args = data_args
    Config.train_args = train_args
    Config.tokenizer = tokenizer


def set_tokenizer(model_args: ModelArguments):
    tokenizer_class = RobertaTokenizer
    # 加载RoBERTa模型的 tokenizer  从预训练模型地址中加载
    tokenizer = tokenizer_class.from_pretrained(
        model_args.model_name_or_path
    )
    tokenizer.add_tokens("-")
    tokenizer.add_special_tokens({"additional_special_tokens": [Config.SEP_TOKEN, Config.NODE_SEP_TOKEN]})
    return tokenizer


def load_datasets(data_args: DataArguments, tokenizer, train_args: TrainingArguments, model_args: ModelArguments):
    data_dir = data_args.data_dir
    max_seq_length = data_args.max_seq_length
    percentage = data_args.data_percentage
    data_cache = data_args.data_cached
    data_overwrite_cache = data_args.data_overwrite_cache

    logger.info(f"Loading Data form {data_dir}\t using cache: {data_cache}\t overate_cache:{data_overwrite_cache}")

    train_data_loader = MyDataLoader(data_dir, "train", percentage, data_overwrite_cache,
                                     data_cache) if train_args.do_train else []

    dev_data_loader = MyDataLoader(data_dir, "dev", 1.0, data_overwrite_cache,
                                   data_cache) if train_args.do_eval else []

    test_data_loader = MyDataLoader(data_dir, "test", 1.0, data_overwrite_cache,
                                    data_cache) if model_args.do_test else []

    logger.info("Finished dataloader")

    train_dataset = MyDataset(train_data_loader, data_dir, "train", max_seq_length, tokenizer, percentage,
                              data_overwrite_cache, data_cache) if train_args.do_train else []
    del train_data_loader
    dev_dataset = MyDataset(dev_data_loader, data_dir, "dev", max_seq_length, tokenizer, 1.0,
                            data_overwrite_cache, data_cache) if train_args.do_eval else []
    del dev_data_loader
    test_dataset = MyDataset(test_data_loader, data_dir, "test", max_seq_length, tokenizer, 1.0,
                             data_overwrite_cache, data_cache) if model_args.do_test else []
    del test_data_loader

    logger.info("Finished dataset")

    logger.info("=" * 50)
    logger.info(f"Train data size:{len(train_dataset)}\t "
                f"dev_dataset:{len(dev_dataset)}\t"
                f"test_dataset:{len(test_dataset)}")
    logger.info("=" * 50)
    logger.info("*" * 50)
    if len(train_dataset) > 0:
        logger.info(f"Train Data Feature example:{train_dataset[0]}")
        logger.info("-" * 50)
    if len(dev_dataset) > 0:
        logger.info(f"Dev Data Feature example:{dev_dataset[0]}")
        logger.info("-" * 50)
    if len(test_dataset) > 0:
        logger.info(f"Test Data Feature example:{test_dataset[0]}")
        logger.info("-" * 50)
    logger.info("*" * 50)
    return train_dataset, dev_dataset, test_dataset


def set_lm_config(model_args: ModelArguments, label_list):
    if model_args.check_points is not None:
        from_model_path = model_args.check_points
    else:
        from_model_path = model_args.model_name_or_path

    config = AutoConfig.from_pretrained(
        from_model_path,
        num_labels=len(label_list),
        finetuning_task="QuestionMatch"
    )
    return config


def train(train_args: TrainingArguments, model_args: ModelArguments, data_args: DataArguments, train_dataset,
          dev_dataset, test_dataset, label_list, lm_config):
    def model_init():
        model = MyMatchModel(lm_config=lm_config,
                             dropout=model_args.dropout,
                             max_seq_length=data_args.max_seq_length,
                             gnn_layer_num=model_args.gnn_layers_num,
                             label_nums=len(label_list))
        return model

    def compute_metrics(eval_pred: EvalPrediction):
        metric = load_metric("utils/accuracy.py")
        logits, labels = eval_pred
        return metric.compute(predictions=logits, references=labels)

    my_trainer = Trainer(
        model_init=model_init,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=None,
    )
    if model_args.do_eval_first:
        results = evaluation(trainer=my_trainer, model_args=model_args)
        logger.info(results)
    if train_args.do_train:
        my_trainer.train(model_path=model_args.model_name_or_path)
        if model_args.check_points is None:
            # 存储模型
            my_trainer.save_model(output_dir=model_args.model_name_or_path)
            logger.info(f"Save model :{model_args.check_points}")
    # if train_args.do_eval:
    #     results = evaluation(trainer=my_trainer, model_args=model_args)
    #     logger.info(results)
    if model_args.do_test:
        results = evaluation_test_dataset(trainer=my_trainer, test_dataset=test_dataset,
                                          model_args=model_args)
        logger.info(results)


def evaluation(trainer, model_args: ModelArguments):
    results = {}
    logger.info("*** Evaluate ***")
    result = trainer.evaluate()
    output_eval_file = os.path.join(model_args.result_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
        results.update(result)
    return result


def evaluation_test_dataset(trainer, model_args, test_dataset):
    logger.info("*** Evaluate Test***")
    # test_dataset = test_dataset
    result = trainer.predict(test_dataset)
    output_eval_file = os.path.join(model_args.result_dir, "text_results.csv")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        writer.write("label, pred\n")
        print(result.metrics)
        # print(type(result.predictions), len(result.predictions), result.predictions)
        # print(type(result.label_ids), len(result.label_ids), result.label_ids)
        for idx in range(len(result.predictions)):
            pred = result.predictions[idx]
            label = result.label_ids[idx]
            print(f"label:{label}\t pred:{pred}")
            writer.write(f"{label}, {pred}\n")
    return result


if __name__ == '__main__':
    model_args, data_args, train_args = parse_args()
    tokenizer = set_tokenizer(model_args)
    set_config(model_args, data_args, train_args, tokenizer)

    train_dataset, dev_dataset, test_dataset = load_datasets(data_args, tokenizer, train_args, model_args)
    label_list = [0, 1]
    lm_config = set_lm_config(model_args, label_list)
    train(train_args, model_args, data_args, train_dataset, dev_dataset, test_dataset, label_list, lm_config)
