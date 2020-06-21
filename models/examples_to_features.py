from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.configuration_bert import BertConfig
from torch.utils.data import DataLoader, TensorDataset
from .multitask_bert import BertForMultitaskLearning
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, input_ids, input_mask, segment_ids,
            text_id, sequence_ids, orig_positions_map
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.text_id = text_id
        self.sequence_ids = sequence_ids
        self.orig_positions_map = orig_positions_map


def convert_examples_to_features_for_bert(
        examples, label2id, max_seq_length, tokenizer, logger
    ):
    """Loads a data file into a list of `InputBatch`s."""

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    neg_sequence_label = '0'
    pad_token = "[PAD]"
    sep_token = "[SEP]"
    cls_token = "[CLS]"

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [cls_token]
        sequence_labels = [neg_sequence_label]
        attention_mask = [1]
        offset = len(tokens)
        orig_positions_map = []

        for i, (token, sequence_label) in enumerate(zip(example.tokens, example.sequence_labels)):
            sub_tokens = tokenizer.tokenize(token)
            num_sub_tokens = len(sub_tokens)

            if offset < max_seq_length:
                additional_offset = num_sub_tokens // 2
                if additional_offset + offset < max_seq_length:
                    orig_positions_map.append(additional_offset + offset)
                else:
                    orig_positions_map.append(offset)
            offset += num_sub_tokens
            tokens += sub_tokens
            sequence_labels += [sequence_label] * num_sub_tokens
            attention_mask += [1] * num_sub_tokens

        tokens.append(sep_token)
        sequence_labels.append(neg_sequence_label)
        attention_mask.append(1)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length - 1] + [sep_token]
            sequence_labels = sequence_labels[:max_seq_length - 1] + [neg_sequence_label]
            attention_mask = attention_mask[:max_seq_length - 1] + [1]
        else:
            num_fit_examples += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = attention_mask
        padding_length = (max_seq_length - len(input_ids))

        input_ids += tokenizer.convert_tokens_to_ids([pad_token]) * padding_length
        input_mask += [0] * padding_length
        segment_ids = [0] * len(input_ids)
        try:
            text_id = label2id['text'][example.text_label]
            sequence_ids = [label2id['sequence'][lab] for lab in sequence_labels]
            sequence_ids += [0] * padding_length
        except KeyError:
            print(label2id['sequence'], sequence_labels)
            raise KeyError

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(sequence_ids) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (text_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("tok_to_pos_map: %s" % " ".join([str(x) for x in orig_positions_map]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("sequence_ids: %s" % " ".join([str(x) for x in sequence_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("text_id: %s (id = %d)" % (example.text_label, text_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                text_id=text_id,
                sequence_ids=sequence_ids,
                orig_positions_map=orig_positions_map
            )
        )
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                       num_fit_examples * 100.0 / len(examples),
                                                                       max_seq_length))
    return features


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


examples_to_features_converters = {
    "bert-large-uncased": convert_examples_to_features_for_bert
}
tokenizers = {
    "bert-large-uncased": BertTokenizer,
    "xlnet-large-cased": XLNetTokenizer
}

models = {
    "bert-large-uncased": BertForMultitaskLearning
}

configs = {
    "bert-large-uncased": BertConfig
}

