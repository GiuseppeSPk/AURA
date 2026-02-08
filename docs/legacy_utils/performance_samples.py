
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import json
import os

# Define Model (Must be identical to Training)
class AURA(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.toxicity_head = nn.Linear(768, 2)
        self.emotion_head = nn.Linear(768, 7)
    def forward(self, input_ids, attention_mask):
        o = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        p = self.dropout(o.pooler_output)
        return self.toxicity_head(p), self.emotion_head(p)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AURA().to(device)
model.load_state_dict(torch.load("outputs/aura_mtl_best.pt", map_location=device))
model.eval()

def get_prediction(text):
    enc = tokenizer.encode_plus(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    with torch.no_grad():
        tox, _ = model(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        pred = torch.argmax(tox, dim=1).item()
    return "OFF" if pred == 1 else "NOT"

print(f"{'Source':<15} | {'True':<5} | {'Pred':<5} | {'Text Preview'}")
print("-" * 80)

results = []

# 1. OLID Samples (Twitter)
olid = pd.read_csv("data/processed/olid_test.csv", encoding='utf-8').sample(5, random_state=42)
for _, row in olid.iterrows():
    text = str(row.get('tweet', row.get('text', '')))
    true = str(row.get('subtask_a', row.get('label', '')))
    true = "OFF" if true in ['OFF', '1', 1] else "NOT"
    pred = get_prediction(text)
    results.append({"Source": "OLID (Twitter)", "True": true, "Pred": pred, "Text": text})

# 2. Jigsaw (Wiki)
with open("data/processed/jigsaw_test.json", encoding='utf-8') as f:
    jigsaw = json.load(f)[:5]
for item in jigsaw:
    text = item['text']
    true = "OFF" if item['label'] in ['OFF', 1] else "NOT"
    pred = get_prediction(text)
    results.append({"Source": "Jigsaw (Wiki)", "True": true, "Pred": pred, "Text": text})

# 3. ToxiGen (AI/Implicit)
with open("data/processed/toxigen_test.json", encoding='utf-8') as f:
    toxigen = json.load(f)[10:15]
for item in toxigen:
    text = item['text']
    true = "OFF" if item['label'] in ['OFF', 1] else "NOT"
    pred = get_prediction(text)
    results.append({"Source": "ToxiGen (AI)", "True": true, "Pred": pred, "Text": text})

df_results = pd.DataFrame(results)
df_results.to_csv("outputs/samples_report.csv", index=False, encoding='utf-8')
print("✅ Report generated: outputs/samples_report.csv")
for _, row in df_results.iterrows():
    print(f"[{row['Source']}] True:{row['True']} | Pred:{row['Pred']} | Text: {row['Text'][:70]}...")
