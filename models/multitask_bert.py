from transformers.modeling_bert import BertPreTrainedModel, BertModel
from transformers.tokenization_bert import BertTokenizer
from torch import nn
from torch.nn import CrossEntropyLoss


class BertForMultitaskLearning(BertPreTrainedModel):
    def __init__(
            self,
            config: BertTokenizer,
            num_sequence_labels: int,
            num_text_labels: int = 2,
            text_clf_weight: float = 1.0,
            sequence_clf_weight: float = 1.0,
            padding_index: int = 100
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
            text_labels=None
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

        text_logits = self.text_classifier(sequence_output)
        sequence_logits = self.sequence_classifier(pooled_output)

        outputs = (sequence_logits,
                   text_logits) + outputs[2:]  # add hidden states and attention if they are here

        if text_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss += self.text_clf_weight * loss_fct(text_logits.view(-1, self.num_text_labels),
                                                    text_labels.view(-1))

        active_loss = attention_mask.view(-1) == 1
        if tag_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.padding_index)
            active_labels = sequence_labels.view(-1)[active_loss]
            active_logits = sequence_logits.view(-1, self.num_sequence_labels)[active_loss]
            loss += self.sequence_clf_weight * loss_fct(active_logits, active_labels)

        if loss > 0:
            outputs = loss

        return outputs
