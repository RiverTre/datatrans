from transformers import BertPreTrainedModel,BertModel
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertCNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.use_cnn = False
        self.AdversarialAttack = True
        if self.use_cnn:
            self.filter_sizes = (5,6,7,12,13,14)                                   # 卷积核尺寸
            self.num_filters = 256                                          # 卷积核数量(channels数)
            self.convs = nn.ModuleList(
                [nn.Conv2d(1, self.num_filters, (k, config.hidden_size)) 
                for k in self.filter_sizes])
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.fc_cnn = nn.Linear(
                self.num_filters * len(self.filter_sizes), 
                config.hidden_size
                )

        self.init_weights()
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    def addAdversarialAttack(self,x):
        r_x = torch.rand_like(x)
        x = x + 0.01*r_x
        return x

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        bert_token_repr = outputs[0]
        pooled_output = outputs[1]
        if self.AdversarialAttack:
            pooled_output = self.addAdversarialAttack(pooled_output)
            bert_token_repr = self.addAdversarialAttack(bert_token_repr)
        pooled_output = self.dropout(pooled_output)
        if self.use_cnn:

            out = bert_token_repr.unsqueeze(1)
            cnn_out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
            cnn_out = self.dropout(cnn_out)
            cnn_out = self.fc_cnn(cnn_out)
            pooled_output = pooled_output + cnn_out
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
