from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer


class InputExample(object):

    def __init__(
            self, guid, tokens, text_label, sequence_labels, text
        ):
        self.guid = guid
        self.tokens = tokens
        self.text_label = text_label
        self.sequence_labels = sequence_labels
        self.text = text


class InputBertFeatures(object):
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

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = ["[CLS]"]
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

        tokens.append("[SEP]")
        sequence_labels.append(neg_sequence_label)
        attention_mask.append(1)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length - 1] + ["[SEP]"]
            sequence_labels = sequence_labels[:max_seq_length - 1] + [neg_tag_label]
            attention_mask = attention_mask[:max_seq_length - 1] + [1]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = attention_mask
        padding = [0] * (max_seq_length - len(input_ids))
        sequence_labels += [neg_sequence_label] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        try:
            text_id = label2id['text'][example.text_label]
            sequence_ids = [label2id['sequence'][lab] for lab in sequence_labels]
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
                logger.info("text_id: %s (id = %d)" % (example.sent_type, text_id))

        features.append(
            InputBertFeatures(
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


def create_examples(dataset, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for example in dataset:
        examples.append(
            InputExample(
                guid=f'{set_type}-{example["idx"]}',
                tokens=example["tokens"],
                text_label=example["text_label"],
                sequence_labels=example["sequence_labels"],
                text=example["text"]
            )
        )
    return examples


examples_to_features_converters = {
    "bert-large-uncased": convert_examples_to_features_for_bert
}
tokenizers = {
    "bert-large-uncased": BertTokenizer,
    "xlnet-large-cased": XLNetTokenizer
}
