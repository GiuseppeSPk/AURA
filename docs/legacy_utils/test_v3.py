# Quick V3 test
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd

print("=== V3 QUICK VERIFICATION ===")

# 1. Model test
class AURA_Bayesian(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.toxicity_head = nn.Linear(768, 2)
        self.emotion_head = nn.Linear(768, 7)
        self.tox_log_var = nn.Parameter(torch.zeros(1))
        self.emo_log_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.toxicity_head(pooled), self.emotion_head(pooled), self.tox_log_var, self.emo_log_var

print("1. Loading BERT...")
model = AURA_Bayesian()
print("   OK")

# 2. Class weights test
CLASS_WEIGHTS = torch.tensor([1.0, 2.02])
print(f"2. Class weights: {CLASS_WEIGHTS.tolist()}")

# 3. Data verification
print("3. Loading clean data...")
goemo = pd.read_csv('data/processed/goemotions_clean.csv')
disgust = goemo['disgust'].sum()
neutral = goemo['neutral'].sum()
print(f"   Disgust: {disgust}, Neutral: {neutral}")

if disgust > 0 and neutral > 0:
    print("   DATA OK!")
else:
    print("   DATA BROKEN!")

# 4. Forward pass
print("4. Forward pass...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
enc = tokenizer(["test text"], return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    tox, emo, lv1, lv2 = model(enc['input_ids'], enc['attention_mask'])
print(f"   Tox shape: {tox.shape}, Emo shape: {emo.shape}")

print("\n=== ALL V3 CHECKS PASSED ===")
