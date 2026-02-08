import nbformat as nbf

print("🔧 Creating SIMPLIFIED BASELINE notebook with unified dataset...")
print("="*60)

# Create new notebook
nb = nbf.v4.new_notebook()

# Cell 1: Title
title = nbf.v4.new_markdown_cell("""# 🧪 AURA BASELINE - Simplified Version

## Professor's Experiment
Testing if RoBERTa can disentangle tasks without explicit architecture.

**Baseline**: Fine-tune on all datasets concatenated → Single toxicity classifier  
**AURA V10**: Task-Specific MHA + Kendall Loss → F1 = 0.7536

**Goal**: Prove Task-Specific architecture adds measurable value.
""")

# Cell 2: Imports & Setup
imports = nbf.v4.new_code_cell("""# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Device: {device}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
""")

# Cell 3: Config
config = nbf.v4.new_code_cell("""# Configuration
CONFIG = {
    'encoder': 'roberta-base',
    'max_length': 128,
    'dropout': 0.3,
    'batch_size': 32,
    'epochs': 15,
    'lr': 2e-5,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'warmup_ratio': 0.1,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    'patience': 5
}

DATA_DIR = './aura-v10-data'
print('📋 Baseline Configuration:')
for k, v in CONFIG.items():
    print(f'   {k}: {v}')
""")

# Cell 4: Dataset Class (Simplified!)
dataset = nbf.v4.new_code_cell("""# Simplified Dataset (uses unified CSV)
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

print('📦 Baseline dataset class defined.')
""")

# Cell 5: Model
model = nbf.v4.new_code_cell("""# Baseline Model: RoBERTa → Linear
class BaselineModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(768, 2)
        
        # Bias init for imbalanced data
        with torch.no_grad():
            self.classifier.bias[0] = 2.5
            self.classifier.bias[1] = -2.5
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

print('🦅 Baseline model defined.')
""")

# Cell 6: Focal Loss
focal = nbf.v4.new_code_cell("""# Focal Loss
def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):
    ce = F.cross_entropy(logits, targets, weight=weight, 
                         reduction='none', label_smoothing=smoothing)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

print('⚖️ Focal loss defined.')
""")

# Cell 7: Load Data
load_data = nbf.v4.new_code_cell("""# Load Data
tokenizer = RobertaTokenizer.from_pretrained(CONFIG['encoder'])

# Load UNIFIED dataset (simplified!)
train_ds = BaselineDataset(f'{DATA_DIR}/unified_baseline_train.csv', 
                           tokenizer, CONFIG['max_length'])
val_ds = BaselineDataset(f'{DATA_DIR}/toxicity_val.csv',
                         tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])

print('📊 Dataset loaded:')
print(f'   Train: {len(train_ds):,} samples (unified)')
print(f'   Val:   {len(val_ds):,} samples (toxicity)')
print(f'   Batches/epoch: {len(train_loader):,}')
""")

# Cell 8: Setup
setup = nbf.v4.new_code_cell("""# Model Setup
model = BaselineModel(CONFIG).to(device)
tox_weights = torch.tensor([0.5, 2.0], device=device)

optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=CONFIG['lr'], 
                              weight_decay=CONFIG['weight_decay'])

total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Training steps: {total_steps:,}')
""")

# Cell 9: Training Functions
train_funcs = nbf.v4.new_code_cell("""# Training Functions
def train_epoch(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=10.0)
    
    for batch in pbar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(ids, mask)
        loss = focal_loss(logits, labels, 
                         gamma=CONFIG['focal_gamma'], 
                         weight=tox_weights, 
                         smoothing=CONFIG['label_smoothing'])
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if len(pbar) % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate():
    model.eval()
    preds, trues = [], []
    for batch in val_loader:
        logits = model(batch['ids'].to(device), batch['mask'].to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch['label'].numpy())
    return f1_score(trues, preds, average='macro', zero_division=0)

print('🎯 Training functions ready.')
""")

# Cell 10: Main Training
main = nbf.v4.new_code_cell("""# Training Loop
print('='*60)
print('🚀 BASELINE TRAINING START')
print('='*60)

best_f1 = 0
patience_counter = 0
history = {'train_loss': [], 'val_f1': []}

for epoch in range(1, CONFIG['epochs'] + 1):
    train_loss = train_epoch(epoch)
    val_f1 = evaluate()
    
    history['train_loss'].append(train_loss)
    history['val_f1'].append(val_f1)
    
    print(f'\\nEpoch {epoch} Summary:')
    print(f'  Train Loss: {train_loss:.4f}')
    print(f'  Val F1:     {val_f1:.4f}')
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'baseline_best.pt')
        print('  >>> BEST MODEL SAVED <<<')
    else:
        patience_counter += 1
        print(f'  (No improvement. Patience: {patience_counter}/{CONFIG[\"patience\"]})')
        
        if patience_counter >= CONFIG['patience']:
            print(f'\\n⚠️ Early stopping at epoch {epoch}')
            break

print('\\n' + '='*60)
print(f'✅ Training Complete. Best Val F1: {best_f1:.4f}')
print('='*60)
""")

# Cell 11: Evaluation & Comparison
eval_compare = nbf.v4.new_code_cell("""# Final Evaluation
model.load_state_dict(torch.load('baseline_best.pt'))
model.eval()

preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch['ids'].to(device), batch['mask'].to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch['label'].numpy())

print('--- Baseline Classification Report ---')
print(classification_report(trues, preds, target_names=['Non-Toxic', 'Toxic']))

# Comparison
print('\\n' + '='*60)
print('📊 BASELINE vs AURA V10')
print('='*60)
baseline_f1 = best_f1
aura_f1 = 0.7536

print(f'Baseline F1:  {baseline_f1:.4f}')
print(f'AURA V10 F1:  {aura_f1:.4f}')
print(f'Difference:   {((aura_f1 - baseline_f1) / baseline_f1 * 100):+.2f}%')
print('='*60)

if aura_f1 > baseline_f1:
    gain = ((aura_f1 - baseline_f1) / baseline_f1 * 100)
    print(f'\\n✅ RESULT: Task-Specific MHA provides {gain:.1f}% improvement!')
    print('   Architecture is JUSTIFIED for the thesis.')
else:
    print('\\n⚠️ Baseline matches or exceeds AURA V10.')
    print('   May need to reconsider architectural complexity.')

# Save history
import json
with open('baseline_history.json', 'w') as f:
    json.dump(history, f, indent=2)
print('\\n📁 History saved to baseline_history.json')
""")

# Add all cells
nb.cells = [title, imports, config, dataset, model, focal, 
            load_data, setup, train_funcs, main, eval_compare]

# Save
output_path = 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_BASELINE_SIMPLIFIED.ipynb'
nbf.write(nb, output_path)

print(f"\n✅ SIMPLIFIED BASELINE notebook created!")
print(f"   File: AURA_BASELINE_SIMPLIFIED.ipynb")
print(f"\n📋 Key improvements:")
print(f"   - Uses unified_baseline_train.csv (single file)")
print(f"   - Simplified dataset class (no task logic)")
print(f"   - Ready for Lightning.ai")
print(f"\n⏰ Estimated training: ~1.5-2 hours")
