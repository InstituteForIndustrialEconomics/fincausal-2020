from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.tokenization_bert import BertTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from itertools import groupby


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self, input_ids, input_mask, segment_ids,
            text_id, sequence_ids, orig_positions_map,
            token_pos_ids=None
        ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.text_id = text_id
        self.sequence_ids = sequence_ids
        self.orig_positions_map = orig_positions_map
        self.token_pos_ids = token_pos_ids


class BertForMultitaskLearning(BertPreTrainedModel):
    def __init__(
            self,
            config: BertTokenizer,
            num_sequence_labels: int,
            num_text_labels: int = 2,
            text_clf_weight: float = 1.0,
            sequence_clf_weight: float = 1.0,
            padding_index: int = 0,
            pooling_type: str = ""
        ):
        super().__init__(config)
        self.text_clf_weight = text_clf_weight
        self.sequence_clf_weight = sequence_clf_weight
        self.num_text_labels = num_text_labels
        self.num_sequence_labels = num_sequence_labels
        self.padding_index = padding_index

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)
        self.text_classifier = nn.Linear(config.hidden_size, self.num_text_labels)

        print(self.num_text_labels, self.num_sequence_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sequence_labels=None,
            text_labels=None,
            token_pos_ids=None
        ):
        loss = 0.0

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        text_logits = self.text_classifier(pooled_output)
        sequence_logits = self.sequence_classifier(sequence_output)

        outputs = (sequence_logits,
                   text_logits) + outputs[2:]  # add hidden states and attention if they are here

        if text_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss += self.text_clf_weight * loss_fct(text_logits.view(-1, self.num_text_labels),
                                                    text_labels.view(-1))

        active_loss = attention_mask.view(-1) == 1
        if sequence_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.padding_index)
            active_labels = sequence_labels.view(-1)[active_loss]
            active_logits = sequence_logits.view(-1, self.num_sequence_labels)[active_loss]
            loss += self.sequence_clf_weight * loss_fct(active_logits, active_labels)

        if loss > 0:
            outputs = loss

        return outputs

    def convert_examples_to_features(
            self, examples, label2id, max_seq_length, tokenizer, logger
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
                    orig_positions_map=orig_positions_map,
                    token_pos_ids=segment_ids
                )
            )
        logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                           num_fit_examples * 100.0 / len(examples),
                                                                           max_seq_length))
        return features


class BertForFakeMultitaskLearning(BertPreTrainedModel):
    def __init__(
            self,
            config: BertTokenizer,
            num_sequence_labels: int,
            num_text_labels: int = 2,
            text_clf_weight: float = 1.0,
            sequence_clf_weight: float = 1.0,
            padding_index: int = 0,
            pooling_type: str = 'first'
        ):
        super().__init__(config)

        self.text_clf_weight = text_clf_weight
        self.sequence_clf_weight = sequence_clf_weight
        self.num_text_labels = num_text_labels
        self.num_sequence_labels = num_sequence_labels
        self.padding_index = padding_index

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sequence_classifier = nn.Linear(config.hidden_size, self.num_sequence_labels)
        self.text_classifier = nn.Linear(config.hidden_size, self.num_text_labels)

        assert pooling_type in ['first', 'avg', 'mid', 'last']
        self.pooling_type = pooling_type

        print(self.num_text_labels, self.num_sequence_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            sequence_labels=None,
            text_labels=None,
            token_pos_ids=None,
            device=torch.device('cuda')
        ):
        loss = 0.0
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]
        valid_sequence_output = self.pool_sequence_outputs(sequence_output, token_pos_ids, device=device)
        valid_sequence_output = self.dropout(valid_sequence_output)

        text_logits = self.text_classifier(pooled_output)
        sequence_logits = self.sequence_classifier(valid_sequence_output)

        outputs = (sequence_logits,
                   text_logits) + outputs[2:]

        if sequence_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            active_labels = sequence_labels.view(-1)
            active_logits = sequence_logits.view(-1, self.num_sequence_labels)
            loss = loss_fct(active_logits, active_labels)

        if loss > 0:
            outputs = loss

        return outputs

    def pool_sequence_outputs(
            self,
            sequence_output,
            token_pos_ids,
            device
        ):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_sequence_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
        if self.pooling_type == 'first':
            for i in range(batch_size):
                prev_pos = -1
                for j in range(max_len):
                    tok_pos = token_pos_ids[i][j].item()
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][tok_pos] = sequence_output[i][j]
                        prev_pos = tok_pos

        elif self.pooling_type == 'avg':
            for i in range(batch_size):
                prev_pos = 0
                token_len = 0
                for j in range(max_len):
                    tok_pos = token_pos_ids[i][j].item()
                    if tok_pos < 0:
                        valid_sequence_output[i][prev_pos] /= token_len
                        break
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][prev_pos] /= token_len
                        valid_sequence_output[i][tok_pos] += sequence_output[i][j]
                        prev_pos = tok_pos
                        token_len = 1
                    else:
                        valid_sequence_output[i][prev_pos] += sequence_output[i][j]
                        token_len += 1

        elif self.pooling_type == 'last':
            for i in range(batch_size):
                prev_pos = 0
                for j in range(max_len + 1):
                    tok_pos = token_pos_ids[i][j].item() if j < max_len else -1
                    if tok_pos != prev_pos:
                        valid_sequence_output[i][prev_pos] = sequence_output[i][j - 1]
                        prev_pos = tok_pos
                    if tok_pos < 0:
                        break

        elif self.pooling_type == 'mid':
            for i in range(batch_size):
                token_positions = [token_pos_ids[i][j].item() for j in range(max_len)]
                positions = []
                for tok_pos, chunk in groupby(token_positions):
                    if tok_pos < 0:
                        break
                    chunk = list(chunk)
                    mid_pos = len(chunk) // 2 + len(positions)
                    if len(chunk) % 2 == 0:
                        mid_pos -= 1
                    positions.extend(chunk)
                    valid_sequence_output[i][tok_pos] = sequence_output[i][mid_pos]

        else:
            raise ValueError

        return valid_sequence_output

    def convert_examples_to_features(
            self, examples, label2id, max_seq_length, tokenizer, logger
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
            token_pos_ids = [0]

            for i, (token, sequence_label) in enumerate(zip(example.tokens, example.sequence_labels)):
                sub_tokens = tokenizer.tokenize(token)
                num_sub_tokens = len(sub_tokens)

                orig_positions_map.append(offset + i)
                tokens += sub_tokens
                token_pos_ids += [offset + i] * num_sub_tokens
                sequence_labels.append(sequence_label)
                attention_mask += [1] * num_sub_tokens

            tokens.append(sep_token)
            sequence_labels.append(neg_sequence_label)
            token_pos_ids.append(token_pos_ids[-1] + 1)
            attention_mask.append(1)
            num_tokens += len(tokens)

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                attention_mask = attention_mask[:max_seq_length]
                token_pos_ids = token_pos_ids[:max_seq_length]
                if len(sequence_labels) > max_seq_length:
                    sequence_labels = sequence_labels[:token_pos_ids[-1] + 1]
            else:
                num_fit_examples += 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = attention_mask
            padding_length = (max_seq_length - len(input_ids))

            input_ids += tokenizer.convert_tokens_to_ids([pad_token]) * padding_length
            input_mask += [0] * padding_length
            segment_ids = [0] * len(input_ids)
            token_pos_ids += [-1] * padding_length
            try:
                text_id = label2id['text'][example.text_label]
                sequence_ids = [label2id['sequence'][lab] for lab in sequence_labels]
                sequence_ids += [0] * (max_seq_length - len(sequence_labels))
            except KeyError:
                print(label2id['sequence'], sequence_labels)
                raise KeyError

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(sequence_ids) == max_seq_length
            assert len(token_pos_ids) == max_seq_length

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
                    logger.info("token_pos_ids: %s" % " ".join([str(x) for x in token_pos_ids]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    text_id=text_id,
                    sequence_ids=sequence_ids,
                    orig_positions_map=orig_positions_map,
                    token_pos_ids=token_pos_ids
                )
            )
        logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
        logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                           num_fit_examples * 100.0 / len(examples),
                                                                           max_seq_length))
        return features
