from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.configuration_bert import BertConfig
from torch.utils.data import DataLoader, TensorDataset
from .multitask_bert import BertForMultitaskLearning, BertForFakeMultitaskLearning
import torch
import os
import json
from collections import Counter


class InputExample(object):

    def __init__(
            self, guid, tokens, text_label,
            sequence_labels, text, task_id
        ):
        self.guid = guid
        self.tokens = tokens
        self.text_label = text_label
        self.sequence_labels = sequence_labels
        self.text = text
        self.task_id = task_id


class DataProcessor(object):
    """Processor for the FINCAUSAL data set."""

    def __init__(self, tag_format: str = "bio", filter_non_causal: bool = False):
        self.tag_format = tag_format
        self.filter_non_causal = filter_non_causal

    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
            if self.filter_non_causal:
                data = [ex for ex in data if str(ex["text_label"]) != "0"]
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"{self.tag_format}_train.json")
            ),
            "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"{self.tag_format}_valid.json")
            ),
            "dev"
        )

    def get_test_examples(self, test_file):
        """See base class."""
        return self.create_examples(
            self._read_json(test_file), "test")

    def get_text_labels(self, data_dir, logger=None):
        """See base class."""
        dataset = self._read_json(os.path.join(data_dir, f"{self.tag_format}_train.json"))
        counter = Counter()
        labels = []
        for example in dataset:
            counter[example['text_label']] += 1
        if logger is not None:
            logger.info(f"text_label: {len(counter)} labels")
        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" % (label, counter * 100.0 / len(dataset)))
            if label not in labels:
                labels.append(label)
        return labels

    def get_sequence_labels(self, data_dir, logger=None):
        """See base class."""
        dataset = self._read_json(os.path.join(data_dir, f"{self.tag_format}_train.json"))
        denominator = len([lab for example in dataset for lab in example['sequence_labels']])
        counter = Counter()
        labels = []
        for example in dataset:
            for lab in example['sequence_labels']:
                counter[lab] += 1
        if logger is not None:
            logger.info(f"sequence_labels: {len(counter)} labels")
        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" % (label, counter * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        return labels

    def create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            examples.append(
                InputExample(
                    guid=f'{set_type}-{example["idx"]}',
                    tokens=example["tokens"],
                    text_label=example["text_label"],
                    sequence_labels=example["sequence_labels"],
                    text=example["text"],
                    task_id=example["task_id"]
                )
            )
        return examples


def get_dataloader_and_text_ids_with_sequence_ids(
        features: list,
        batch_size: int
    ):
    input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long
    )
    segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    text_labels_ids = torch.tensor(
        [f.text_id for f in features], dtype=torch.long
    )
    sequence_labels_ids = torch.tensor(
        [f.sequence_ids for f in features], dtype=torch.long
    )

    eval_data = TensorDataset(
        input_ids, input_mask, segment_ids,
        text_labels_ids, sequence_labels_ids
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, text_labels_ids, sequence_labels_ids

tokenizers = {
    "bert-large-uncased": BertTokenizer,
    "xlnet-large-cased": XLNetTokenizer
}

models = {
    "bert-large-uncased": BertForMultitaskLearning,
    "bert-large-uncased-fake": BertForFakeMultitaskLearning
}

configs = {
    "bert-large-uncased": BertConfig
}

