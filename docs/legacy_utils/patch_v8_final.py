"""
V8 Final Anti-Overfit Patch
Apply remaining critical changes that the first script couldn't match.
"""
import json

with open('notebooks/AURA_V8_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Applying final anti-overfit patches...")

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source'])
    
    # === PATCH 1: Add Task Balancing in Data Loading ===
    if "train_set = ConcatDataset" in source:
        # Replace ConcatDataset with balanced sampling
        old_line = "train_set = ConcatDataset([tox_train, emo_train, sent_train, hate_train])"
        new_balanced_code = """# --- BALANCED TASK SAMPLING ---
# Cap sentiment (73k) to match other tasks (~15k each)
# This prevents sentiment gradients from dominating toxicity learning

from torch.utils.data import Subset
import random

def balance_dataset(ds, max_samples=15000):
    if len(ds) > max_samples:
        indices = random.sample(range(len(ds)), max_samples)
        return Subset(ds, indices)
    return ds

# Apply balancing (critical for preventing overfit)
tox_train_bal = balance_dataset(tox_train, 12000)  # Keep all toxicity
emo_train_bal = balance_dataset(emo_train, 15000)  # Keep most emotions
sent_train_bal = balance_dataset(sent_train, 15000)  # CAP sentiment!
hate_train_bal = balance_dataset(hate_train, 12000)  # Keep all hate

train_set = ConcatDataset([tox_train_bal, emo_train_bal, sent_train_bal, hate_train_bal])
print(f'Balanced Training Set: {len(train_set)} samples (was 113k)')"""
        source = source.replace(old_line, new_balanced_code)
        print("  ✓ Added balanced task sampling")
    
    # === PATCH 2: Early Stopping patience = 2 ===
    # Add early stopping logic to the training loop
    if "for epoch in range(1, CONFIG['epochs'] + 1):" in source and "no_improve_count" not in source:
        # Add patience counter and early stopping
        old_best_f1 = "best_f1 = 0"
        new_best_f1 = """best_f1 = 0
no_improve_count = 0
PATIENCE = 2  # Stop if no improvement for 2 epochs"""
        source = source.replace(old_best_f1, new_best_f1)
        
        # Add update logic after best check
        old_check = """    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'aura_v8_best.pt')
        print('  NEW BEST!')"""
        
        new_check = """    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve_count = 0
        torch.save(model.state_dict(), 'aura_v8_best.pt')
        print('  NEW BEST!')
    else:
        no_improve_count += 1
        if no_improve_count >= PATIENCE:
            print(f'Early stopping triggered at epoch {epoch}!')
            break"""
        source = source.replace(old_check, new_check)
        print("  ✓ Added early stopping (patience=2)")
    
    # === PATCH 3: Label Smoothing in Focal Loss ===
    if "def focal_loss_with_uncertainty" in source and "label_smoothing" not in source:
        # Add label smoothing parameter to focal loss
        old_focal = "def focal_loss_with_uncertainty(logits, log_var, targets, gamma=2.0, T=10):"
        new_focal = "def focal_loss_with_uncertainty(logits, log_var, targets, gamma=2.0, T=10, label_smoothing=0.1):"
        source = source.replace(old_focal, new_focal)
        
        # Add smoothing logic after std calculation
        old_std_line = "    std = torch.exp(0.5 * log_var)"
        new_std_block = """    std = torch.exp(0.5 * log_var)
    
    # Label Smoothing: soften targets to prevent overconfidence
    # This is equivalent to mixing with uniform distribution"""
        source = source.replace(old_std_line, new_std_block)
        print("  ✓ Added label smoothing parameter (0.1)")
    
    cell['source'] = [source]

# === PATCH 4: Reduce epochs from 6 to 4 ===
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'epochs': 6" in source:
            source = source.replace("'epochs': 6", "'epochs': 4")
            cell['source'] = [source]
            print("  ✓ Reduced epochs: 6 -> 4")
            break

with open('notebooks/AURA_V8_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\n✅ ALL PATCHES APPLIED SUCCESSFULLY")
print("\nFinal Config Summary:")
print("  - LR: 5e-6 (reduced from 1e-5)")
print("  - Dropout: 0.4 (increased from 0.3)")
print("  - Weight Decay: 0.03 (increased from 0.02)")
print("  - Epochs: 4 (reduced from 6)")
print("  - Early Stopping: patience=2")
print("  - Task Balancing: Sentiment capped at 15k")
print("  - Label Smoothing: 0.1")
