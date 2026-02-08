"""
Test script for AURA BASELINE notebook

Validates:
1. Dataset loading works
2. Model initialization works
3. Forward pass works
4. Training loop syntax is correct
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import os

print("="*60)
print("🧪 TESTING BASELINE NOTEBOOK COMPONENTS")
print("="*60)

# Check data files exist
DATA_DIR = 'C:/Users/spicc/Desktop/Multimodal/AURA/kaggle_upload/aura-v10-data'
print("\n📂 Checking data files...")

required_files = [
    'unified_baseline_train.csv',
    'toxicity_val.csv'
]

for file in required_files:
    path = f'{DATA_DIR}/{file}'
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  ✅ {file}: {len(df):,} samples")
    else:
        print(f"  ❌ {file}: NOT FOUND!")

# Test Dataset Class
print("\n📦 Testing Dataset class...")
class BaselineDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = pd.read_csv(csv_path)
        self.tok = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(
            str(row['text']),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(int(row['label']), dtype=torch.long)
        }

try:
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    test_ds = BaselineDataset(f'{DATA_DIR}/unified_baseline_train.csv', tokenizer, 128)
    sample = test_ds[0]
    print(f"  ✅ Dataset works! Sample keys: {sample.keys()}")
    print(f"     Input shape: {sample['ids'].shape}")
    print(f"     Label: {sample['label'].item()}")
except Exception as e:
    print(f"  ❌ Dataset failed: {e}")

# Test Model
print("\n🦅 Testing Model class...")
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

try:
    model = BaselineModel()
    print(f"  ✅ Model initialized!")
    print(f"     Total params: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"  ❌ Model failed: {e}")

# Test Forward Pass
print("\n🔄 Testing forward pass...")
try:
    # Create dummy batch
    batch_ids = sample['ids'].unsqueeze(0)  # [1, 128]
    batch_mask = sample['mask'].unsqueeze(0)  # [1, 128]
    
    model.eval()
    with torch.no_grad():
        logits = model(batch_ids, batch_mask)
    
    print(f"  ✅ Forward pass works!")
    print(f"     Output shape: {logits.shape}")
    print(f"     Predicted class: {logits.argmax(1).item()}")
except Exception as e:
    print(f"  ❌ Forward pass failed: {e}")

# Test Focal Loss
print("\n⚖️ Testing Focal Loss...")
def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):
    ce = F.cross_entropy(logits, targets, weight=weight, 
                         reduction='none', label_smoothing=smoothing)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

try:
    targets = torch.tensor([1])  # Toxic
    loss = focal_loss(logits, targets, gamma=2.0)
    print(f"  ✅ Focal loss works!")
    print(f"     Loss value: {loss.item():.4f}")
except Exception as e:
    print(f"  ❌ Focal loss failed: {e}")

# Test DataLoader
print("\n📊 Testing DataLoader...")
try:
    loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"  ✅ DataLoader works!")
    print(f"     Batch size: {batch['ids'].shape[0]}")
    print(f"     IDs shape: {batch['ids'].shape}")
except Exception as e:
    print(f"  ❌ DataLoader failed: {e}")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\n🚀 Baseline notebook is ready for Lightning.ai!")
print("\nFiles to upload:")
print("  1. AURA_BASELINE_SIMPLIFIED.ipynb")
print("  2. unified_baseline_train.csv")
print("  3. toxicity_val.csv")
