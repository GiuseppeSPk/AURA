import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# CONFIG V10 MINI
CONFIG = {
    'encoder': 'bert-base-uncased',
    'max_length': 32, # Short for speed
    'num_emotion_classes': 7,
    'dropout': 0.5,
    'batch_size': 4,
    'epochs': 2, # 1 Frozen, 1 Unfrozen
    'freezing_epochs': 1
}

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, length=10):
        self.len = length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self): return self.len
    def __getitem__(self, idx):
        enc = self.tokenizer("hello world", max_length=32, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'tox_label': torch.tensor(1),
            'emo_label': torch.zeros(7),
            'sent_label': torch.tensor(1),
            'task': 'toxicity',
            'task_mask_tox': torch.tensor(True),
            'task_mask_emo': torch.tensor(False),
            'task_mask_sent': torch.tensor(False)
        }

# Model
class AURA_CERBERUS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.toxicity_head = nn.Linear(768, 2)
        self.emotion_head = nn.Linear(768, 7)
        self.sentiment_head = nn.Linear(768, 2)
        self.log_var_tox = nn.Parameter(torch.tensor(0.0))
        self.log_var_emo = nn.Parameter(torch.tensor(0.0))
        self.log_var_sent = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        pooled = self.dropout(out.pooler_output)
        return {
            'toxicity': self.toxicity_head(pooled),
            'emotion': self.emotion_head(pooled),
            'sentiment': self.sentiment_head(pooled),
            'log_var_tox': self.log_var_tox,
            'log_var_emo': self.log_var_emo,
            'log_var_sent': self.log_var_sent
        }

def run():
    print("Initializing Dry Run V10...")
    model = AURA_CERBERUS(CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loader = DataLoader(DummyDataset(), batch_size=4)
    
    for epoch in range(1, 3):
        # FREEZING LOGIC CHECK
        if epoch <= CONFIG['freezing_epochs']:
            print(f"Epoch {epoch}: FREEZING BERT.")
            for param in model.bert.parameters():
                param.requires_grad = False
            # Verify
            assert model.bert.pooler.dense.weight.requires_grad == False
            print(">> Verification: BERT is successfully locked.")
        else:
            print(f"Epoch {epoch}: UNFREEZING BERT.")
            for param in model.bert.parameters():
                param.requires_grad = True
            # Verify
            assert model.bert.pooler.dense.weight.requires_grad == True
            print(">> Verification: BERT is successfully unlocked.")
            
        # Simulating Train Step
        model.train()
        batch = next(iter(loader))
        out = model(batch['input_ids'], batch['attention_mask'])
        loss = out['toxicity'].sum() # Dummy loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch} Step completed.")
        
    print("\nSUCCESS: V10 Logic Check Passed.")

if __name__ == '__main__':
    run()
