"""设计意图识别的模型类:基于Bert的多任务分类模型"""
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from src.schema.config import get_config

class BertIntentModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.intent_classifier = nn.Linear(config.hidden_size, len(get_config().intent_labels)) # 主任务
        self.human_classifier = nn.Linear(config.hidden_size, 2)  #次任务
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, human_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = self.dropout(outputs.pooler_output)
        intent_logits = self.intent_classifier(pooled)
        human_logits = self.human_classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(intent_logits, labels)
            if human_labels is not None:
                loss += 0.3 * nn.functional.cross_entropy(human_logits, human_labels)

        return {"loss": loss, "logits": intent_logits, "human_logits": human_logits}