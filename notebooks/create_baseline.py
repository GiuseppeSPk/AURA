import nbformat as nbf

print("🔬 Creating BASELINE notebook for Professor's experiment...")
print("="*60)

# Create new notebook
nb = nbf.v4.new_notebook()

# Cell 1: Title
title = nbf.v4.new_markdown_cell("""# 🧪 AURA BASELINE - Professor's Control Experiment

## Hypothesis to Test
**Professor's Counter-Hypothesis**: 
> *"A large enough model could disentangle the four channels by virtue of training only"*

## Baseline Architecture
- **Model**: RoBERTa-base (standard, no modifications)
- **Training**: Fine-tune on concatenation of all 4 datasets
- **No Task-Specific Attention**
- **No Kendall Loss** (simple cross-entropy)

## Goal
Compare with AURA V10 to prove Task-Specific MHA adds value.

**Expected**: Baseline F1 < AURA V10 F1 (0.7536)  
**If True**: Task-specific architecture is justified ✅
""")

# Cell 2: Imports
imports = nbf.v4.new_code_cell("""# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
config = nbf.v4.new_code_cell("""# Configuration (Same as AURA V10 where applicable)
CONFIG = {
    'encoder': 'roberta-base',
    'max_length': 128,
    'dropout': 0.3,
    'batch_size': 32,  # L40S can handle it
    'epochs': 15,      # Enough for convergence
    'lr': 2e-5,        # Standard BERT fine-tuning LR
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'warmup_ratio': 0.1,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    'patience': 5
}

DATA_DIR = './aura-v10-data'  # Lightning.ai path
print('📋 BASELINE Configuration:')
for k, v in CONFIG.items():
    print(f'   {k}: {v}')
""")

# Cell 4: Dataset Classes
dataset_classes = nbf.v4.new_code_cell("""# Unified Dataset (All tasks treated as Toxicity binary)
class UnifiedDataset(Dataset):
    \"\"\"Concatenate all datasets and use only toxicity labels.
    
    This is the BASELINE approach: expose RoBERTa to diverse data
    but without task-specific architecture.
    \"\"\"
    def __init__(self, path, tokenizer, max_len, task_name):
        self.df = pd.read_csv(path)
        self.tok = tokenizer
        self.max_len = max_len
        self.task = task_name
        
    def __len__(self):
        return len(self.df)
    
    def encode(self, text):
        return self.tok(
            str(text),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.encode(row['text'])
        
        # Extract toxicity label (0 or 1)
        if self.task == 'toxicity':
            label = int(row['label'])
        elif self.task == 'emotion':
            # For emotion, consider ANY emotion as "non-toxic" (label=0)
            label = 0
        elif self.task == 'sentiment':
            # Sentiment: negative=0 (non-toxic), positive=0 (non-toxic)
            label = 0
        elif self.task == 'reporting':
            # Reporting: all are non-toxic examples
            label = 0
        
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

print('📦 Unified dataset class defined.')
""")

# Cell 5: Model (Simple RoBERTa)
model_def = nbf.v4.new_code_cell("""# BASELINE Model: Standard RoBERTa + Linear Classifier
class BaselineModel(nn.Module):
    \"\"\"Simple RoBERTa → [CLS] → Linear(768, 2)
    
    NO task-specific attention
    NO Kendall weighting
    Just pure fine-tuning on concatenated data
    \"\"\"
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(config['encoder'])
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(768, 2)  # Binary: Non-Toxic / Toxic
        
        # Bias initialization for imbalanced data
        with torch.no_grad():
            self.classifier.bias[0] = 2.5   # Non-Toxic prior
            self.classifier.bias[1] = -2.5  # Toxic prior
    
    def forward(self, input_ids, attention_mask):
        # Get [CLS] token representation
        outputs = self.roberta(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        
        # Classify
        logits = self.classifier(self.dropout(cls_output))  # [batch, 2]
        return logits

print('🦅 Baseline model defined.')
""")

# Cell 6: Focal Loss
focal_loss = nbf.v4.new_code_cell("""# Focal Loss (Same as AURA V10)
def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):
    \"\"\"Focal Loss for handling class imbalance.\"\"\"
    ce = F.cross_entropy(logits, targets, weight=weight, 
                         reduction='none', label_smoothing=smoothing)
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

print('⚖️ Focal loss defined.')
""")

# Cell 7: Load Data
load_data = nbf.v4.new_code_cell("""# Load All Datasets and Concatenate
tokenizer = RobertaTokenizer.from_pretrained(CONFIG['encoder'])

# Load each dataset
tox_train = UnifiedDataset(f'{DATA_DIR}/toxicity_train.csv', tokenizer, 
                           CONFIG['max_length'], 'toxicity')
emo_train = UnifiedDataset(f'{DATA_DIR}/emotions_train.csv', tokenizer, 
                           CONFIG['max_length'], 'emotion')
sent_train = UnifiedDataset(f'{DATA_DIR}/sentiment_train.csv', tokenizer, 
                            CONFIG['max_length'], 'sentiment')
rep_train = UnifiedDataset(f'{DATA_DIR}/reporting_examples.csv', tokenizer, 
                           CONFIG['max_length'], 'reporting')
tox_val = UnifiedDataset(f'{DATA_DIR}/toxicity_val.csv', tokenizer, 
                         CONFIG['max_length'], 'toxicity')

# CONCATENATE all training data (Professor's baseline approach)
train_ds = ConcatDataset([tox_train, emo_train, sent_train, rep_train])

# DataLoaders
train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(tox_val, batch_size=CONFIG['batch_size'])

print('='*60)
print('📊 BASELINE DATASET')
print('='*60)
print(f'Training: {len(train_ds):,} samples (concatenated)')
print(f'  - Toxicity:  {len(tox_train):,}')
print(f'  - Emotion:   {len(emo_train):,}')
print(f'  - Sentiment: {len(sent_train):,}')
print(f'  - Reporting: {len(rep_train):,}')
print(f'Validation: {len(tox_val):,} (Toxicity only)')
""")

# Cell 8: Setup Training
setup = nbf.v4.new_code_cell("""# Model and Optimizer Setup
model = BaselineModel(CONFIG).to(device)
tox_weights = torch.tensor([0.5, 2.0], device=device)  # Class weights

# Optimizer (same as AURA V10 encoder LR)
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=CONFIG['lr'], 
                              weight_decay=CONFIG['weight_decay'])

# Scheduler with warmup
total_steps = len(train_loader) * CONFIG['epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Optimization steps: {total_steps:,}')
""")

# Cell 9: Training Loop
training = nbf.v4.new_code_cell("""# Training Loop
def train_epoch(epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        logits = model(ids, mask)
        loss = focal_loss(logits, labels, 
                         gamma=CONFIG['focal_gamma'], 
                         weight=tox_weights, 
                         smoothing=CONFIG['label_smoothing'])
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
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
main_train = nbf.v4.new_code_cell("""# BASELINE Training
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
    
    print(f'\\nEpoch {epoch}:')
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
print(f'✅ BASELINE Training Complete')
print(f'🏆 Best Val F1: {best_f1:.4f}')
print('='*60)
""")

# Cell 11: Final Evaluation
final_eval = nbf.v4.new_code_cell("""# Final Evaluation & Comparison
model.load_state_dict(torch.load('baseline_best.pt'))
model.eval()

preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch['ids'].to(device), batch['mask'].to(device))
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(batch['label'].numpy())

print('--- BASELINE Classification Report ---')
print(classification_report(trues, preds, target_names=['Non-Toxic', 'Toxic']))

# Comparison with AURA V10
print('\\n' + '='*60)
print('📊 COMPARISON: BASELINE vs AURA V10')
print('='*60)
baseline_f1 = best_f1
aura_f1 = 0.7536  # From AURA V10 run

print(f'BASELINE F1:  {baseline_f1:.4f}')
print(f'AURA V10 F1:  {aura_f1:.4f}')
print(f'Improvement:  {((aura_f1 - baseline_f1) / baseline_f1 * 100):+.2f}%')
print('='*60)

if aura_f1 > baseline_f1:
    print('✅ CONCLUSION: Task-Specific MHA architecture is JUSTIFIED!')
    print('   The explicit feature disentanglement provides measurable benefit.')
else:
    print('⚠️ Professor was right: RoBERTa can disentangle without explicit architecture.')
""")

# Add all cells
nb.cells = [title, imports, config, dataset_classes, model_def, focal_loss, 
            load_data, setup, training, main_train, final_eval]

# Save
output_path = 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_BASELINE.ipynb'
nbf.write(nb, output_path)

print(f"\n✅ BASELINE notebook created!")
print(f"   Saved to: AURA_BASELINE.ipynb")
print(f"\n🔬 What it does:")
print(f"   - Trains RoBERTa on concatenated datasets (NO task-specific arch)")
print(f"   - Uses same hyperparameters as AURA V10 for fair comparison")
print(f"   - Evaluates on Toxicity validation set")
print(f"   - Automatically compares with AURA V10 (F1 = 0.7536)")
print(f"\n⏰ Estimated training time: ~1.5-2 hours on L40S")
print(f"\n🎯 Upload to Lightning.ai and run to validate your architecture!")
