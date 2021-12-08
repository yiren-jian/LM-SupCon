"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
from transformers.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead
from transformers.modeling_outputs import SequenceClassifierOutput

import logging
logger = logging.getLogger(__name__)


##### added for contrastive learning
class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, hidden_size=768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x


def resize_token_type_embeddings(model, new_num_types: int, random_segment: bool):
    """
    Resize the segment (token type) embeddings for BERT
    """
    if hasattr(model, 'bert'):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, 'bert'):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.mlp = MLPLayer()
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        encoder_forward=False,
    ):
        if encoder_forward is False:
            batch_size = input_ids.size(0)

            if mask_pos is not None:
                mask_pos = mask_pos.squeeze()

            # Encode everything
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Get <mask> token representation
            sequence_output, pooled_output = outputs[:2]
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            # Logits over vocabulary tokens
            prediction_mask_scores = self.cls(sequence_mask_output)

            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits = logsoftmax(logits) # Log prob of right polarity

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.KLDivLoss(log_target=True)
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss = loss_fct(logits.view(-1, 2), labels)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            output = (logits,)
            if self.num_labels == 1:
                # Regression output
                output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
            return ((loss,) + output) if loss is not None else output

        else:
            batch_size = input_ids.size(0)

            if mask_pos is not None:
                mask_pos = mask_pos.squeeze()

            # Encode everything
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # Get <mask> token representation
            sequence_output, pooled_output = outputs[:2]
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            return self.mlp(sequence_mask_output)



class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.mlp = MLPLayer()
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        encoder_forward=False,
    ):
        if encoder_forward is False:
            batch_size = input_ids.size(0)

            if mask_pos is not None:
                mask_pos = mask_pos.squeeze()

            # Encode everything
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )

            # Get <mask> token representation
            sequence_output, pooled_output = outputs[:2]
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            # Logits over vocabulary tokens
            prediction_mask_scores = self.lm_head(sequence_mask_output)

            # Exit early and only return mask logits.
            if self.return_full_softmax:
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            # Return logits for each label
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

            # Regression task
            if self.config.num_labels == 1:
                logsoftmax = nn.LogSoftmax(-1)
                logits = logsoftmax(logits) # Log prob of right polarity

            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression task
                    loss_fct = nn.KLDivLoss(log_target=True)
                    labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                    loss = loss_fct(logits.view(-1, 2), labels)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            output = (logits,)
            if self.num_labels == 1:
                # Regression output
                output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
            return ((loss,) + output) if loss is not None else output

        else: #### encoder_forward is True
            batch_size = input_ids.size(0)

            if mask_pos is not None:
                mask_pos = mask_pos.squeeze()

            # Encode everything
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )

            # Get [CLS] token representation
            sequence_output, pooled_output = outputs[:2]
            sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

            # return self.mlp(sequence_output[:, 0])   ### [bsz, hidden_dim]
            return self.mlp(sequence_mask_output)
