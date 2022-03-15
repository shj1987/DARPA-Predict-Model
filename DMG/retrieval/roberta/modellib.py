import torch
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, RobertaModel
from transformers.modeling_roberta import RobertaClassificationHead


class RobertaForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.

    https://github.com/ThilinaRajapakse/simpletransformers/blob/master/simpletransformers/custom_models/models.py
    """

    def __init__(self, config):
        super(RobertaForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = len(config.id2label)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)  # , output_all_encoded_layers=False)
        logits = self.classifier(outputs[0])
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
        
    def freeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        for param in self.roberta.parameters():
            param.requires_grad = True
