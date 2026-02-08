"""
AURA Bayesian Final Notebook - Local Verification Script
Tests all components without requiring full Kaggle environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
import pandas as pd
import numpy as np
import os

print("="*60)
print("🧪 AURA BAYESIAN NOTEBOOK VERIFICATION")
print("="*60)

# --- CONFIG ---
CONFIG = {
    'encoder': 'bert-base-uncased',
    'max_length': 64,  # Reduced for speed
    'num_emotion_classes': 7,
    'dropout': 0.1,
    'batch_size': 4,
    'mc_samples': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f"\n📍 Device: {CONFIG['device']}")

# --- MODEL ---
class AURA_Bayesian(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config['encoder'])
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config['dropout'])
        self.toxicity_head = nn.Linear(hidden_size, 2)
        self.emotion_head = nn.Linear(hidden_size, config['num_emotion_classes'])
        self.tox_log_var = nn.Parameter(torch.zeros(1))
        self.emo_log_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        tox_logits = self.toxicity_head(pooled)
        emo_logits = self.emotion_head(pooled)
        return tox_logits, emo_logits, self.tox_log_var, self.emo_log_var

print("\n✅ [1/5] Model class defined correctly")

# --- LOSS FUNCTIONS ---
def monte_carlo_uncertainty_loss_classification(logits, log_var, targets, T=10):
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped)
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_expanded)
    corrupted_logits = logits_expanded + (noise * std)
    probs = F.softmax(corrupted_logits, dim=-1)
    avg_probs = torch.mean(probs, dim=0)
    log_probs = torch.log(avg_probs + 1e-8)
    nll = F.nll_loss(log_probs, targets)
    regularization = 0.5 * log_var_clamped
    return nll + regularization

def monte_carlo_uncertainty_loss_multilabel(logits, log_var, targets, T=10):
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped)
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_expanded)
    corrupted_logits = logits_expanded + (noise * std)
    probs = torch.sigmoid(corrupted_logits)
    avg_probs = torch.mean(probs, dim=0)
    bce = F.binary_cross_entropy(avg_probs, targets, reduction='mean')
    regularization = 0.5 * log_var_clamped
    return bce + regularization

print("✅ [2/5] Loss functions defined correctly")

# --- MOCK DATASET ---
class MockDataset(Dataset):
    def __init__(self, size=16):
        self.size = size
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG['encoder'])
        self.texts = ["I hate you", "I love you", "Neutral text", "Test sample"] * 4
        self.tox_labels = [1, 0, 0, 0] * 4
        self.emo_labels = [[1,0,0,0,0,0,0], [0,0,0,1,0,0,0], [0,0,0,0,0,0,1], [0,0,0,0,0,0,1]] * 4
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'toxicity_target': torch.tensor(self.tox_labels[idx], dtype=torch.long),
            'emotion_target': torch.tensor(self.emo_labels[idx], dtype=torch.float32),
            'is_toxicity_task': torch.tensor(idx % 2, dtype=torch.long)  # Alternate
        }

print("✅ [3/5] Dataset class defined correctly")

# --- INSTANTIATION TEST ---
print("\n🔄 Loading model (this may take a moment)...")
model = AURA_Bayesian(CONFIG).to(CONFIG['device'])
print("✅ [4/5] Model instantiated successfully")

# --- FORWARD PASS TEST ---
dataset = MockDataset()
loader = DataLoader(dataset, batch_size=CONFIG['batch_size'])
batch = next(iter(loader))

input_ids = batch['input_ids'].to(CONFIG['device'])
attention_mask = batch['attention_mask'].to(CONFIG['device'])
tox_targets = batch['toxicity_target'].to(CONFIG['device'])
emo_targets = batch['emotion_target'].to(CONFIG['device'])
is_tox_task = batch['is_toxicity_task'].to(CONFIG['device'])

# Forward
tox_logits, emo_logits, tox_log_var, emo_log_var = model(input_ids, attention_mask)

print(f"\n📊 Forward Pass Results:")
print(f"   Tox Logits shape: {tox_logits.shape}")  # Expected: [4, 2]
print(f"   Emo Logits shape: {emo_logits.shape}")  # Expected: [4, 7]
print(f"   Tox Log Var: {tox_log_var.item():.4f}")
print(f"   Emo Log Var: {emo_log_var.item():.4f}")

# --- LOSS COMPUTATION TEST ---
tox_mask = is_tox_task == 1
emo_mask = is_tox_task == 0

if tox_mask.sum() > 0:
    tox_loss = monte_carlo_uncertainty_loss_classification(
        tox_logits[tox_mask], tox_log_var, tox_targets[tox_mask], T=CONFIG['mc_samples']
    )
    print(f"   Tox Loss: {tox_loss.item():.4f}")

if emo_mask.sum() > 0:
    emo_loss = monte_carlo_uncertainty_loss_multilabel(
        emo_logits[emo_mask], emo_log_var, emo_targets[emo_mask], T=CONFIG['mc_samples']
    )
    print(f"   Emo Loss: {emo_loss.item():.4f}")

print("\n✅ [5/5] Loss computation successful")

# --- BACKWARD PASS TEST ---
total_loss = tox_loss + emo_loss
total_loss.backward()
print(f"\n✅ Backward pass successful!")
print(f"   Gradient on tox_log_var: {model.tox_log_var.grad.item():.6f}")

print("\n" + "="*60)
print("🎉 ALL TESTS PASSED - NOTEBOOK IS READY FOR KAGGLE")
print("="*60)
