import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import yaml

class AURA(nn.Module):
    def __init__(self, config_path="config.yaml"):
        super(AURA, self).__init__()
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.bert = BertModel.from_pretrained(self.config['model']['encoder'])
        self.dropout = nn.Dropout(self.config['model']['dropout'])
        
        # Head A: Toxicity (2 classes)
        self.toxicity_head = nn.Linear(self.config['model']['hidden_size'], self.config['model']['toxicity_classes'])
        
        # Head B: Emotion (Now 7 classes if configured so)
        num_emotion_classes = self.config['data']['num_emotion_classes']
        self.emotion_head = nn.Linear(self.config['model']['hidden_size'], num_emotion_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        toxicity_logits = self.toxicity_head(pooled_output)
        emotion_logits = self.emotion_head(pooled_output)
        
        return toxicity_logits, emotion_logits
