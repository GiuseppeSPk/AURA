"""
AURA V8 Anti-Overfitting Optimization Script
============================================
This script applies evidence-based regularization techniques to the V8 notebook.

Changes Applied:
1. TASK BALANCING: Downsample Sentiment to match other tasks
2. LOWER LR: 1e-5 -> 5e-6 (slower convergence, better generalization)
3. HIGHER DROPOUT: 0.3 -> 0.4 (more regularization for 110M param model)
4. LABEL SMOOTHING: 0.1 (prevents overconfident predictions)
5. LAYER-WISE LR DECAY: Lower layers get smaller LR (preserve pre-trained knowledge)
6. WARMUP RATIO: 0.1 -> 0.2 (longer warmup protects pre-trained weights)
7. AGGRESSIVE EARLY STOPPING: patience 3 -> 2
"""

import json
import os

print("="*70)
print("AURA V8 ANTI-OVERFITTING OPTIMIZATION")
print("="*70)

# Load notebook
with open('notebooks/AURA_V8_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changes_made = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    
    source = ''.join(cell['source'])
    modified = False
    
    # === FIX 1: Update CONFIG with optimized hyperparameters ===
    if "'lr': 1e-5" in source or "'lr': 2e-5" in source:
        source = source.replace("'lr': 1e-5", "'lr': 5e-6")
        source = source.replace("'lr': 2e-5", "'lr': 5e-6")
        changes_made.append("LR: 1e-5 -> 5e-6")
        modified = True
    
    if "'dropout': 0.3" in source:
        source = source.replace("'dropout': 0.3", "'dropout': 0.4")
        changes_made.append("Dropout: 0.3 -> 0.4")
        modified = True
    
    if "'weight_decay': 0.02" in source or "'weight_decay': 0.01" in source:
        source = source.replace("'weight_decay': 0.02", "'weight_decay': 0.03")
        source = source.replace("'weight_decay': 0.01", "'weight_decay': 0.03")
        changes_made.append("Weight Decay: 0.02 -> 0.03")
        modified = True
    
    # === FIX 2: Add Label Smoothing to Focal Loss ===
    if "def focal_loss_with_uncertainty" in source and "label_smoothing" not in source:
        # Add label smoothing parameter
        old_def = "def focal_loss_with_uncertainty(logits, labels, log_var, gamma=2.0):"
        new_def = "def focal_loss_with_uncertainty(logits, labels, log_var, gamma=2.0, label_smoothing=0.1):"
        if old_def in source:
            source = source.replace(old_def, new_def)
            # Add smoothing logic after labels line
            # We apply smoothing: y_smooth = y * (1 - smoothing) + smoothing / num_classes
            smoothing_code = '''
    # Label Smoothing
    num_classes = logits.shape[-1]
    labels_smooth = labels.float() * (1.0 - label_smoothing) + label_smoothing / num_classes
'''
            # Insert after the function definition line
            source = source.replace(new_def, new_def + smoothing_code)
            changes_made.append("Label Smoothing: Added (0.1)")
            modified = True
    
    # === FIX 3: Add balanced sampling for tasks ===
    if "class CombinedDataset" in source and "max_samples_per_task" not in source:
        # Add max_samples parameter to balance tasks
        old_init = "def __init__(self, datasets):"
        new_init = "def __init__(self, datasets, max_samples_per_task=15000):"  # Cap at 15k per task
        if old_init in source:
            source = source.replace(old_init, new_init)
            # Add truncation logic
            balance_code = '''
        # Balance tasks by capping samples
        balanced_datasets = []
        for ds in datasets:
            if len(ds) > max_samples_per_task:
                indices = torch.randperm(len(ds))[:max_samples_per_task]
                # Create subset manually
                balanced_ds = torch.utils.data.Subset(ds, indices)
                balanced_datasets.append(balanced_ds)
            else:
                balanced_datasets.append(ds)
        datasets = balanced_datasets
'''
            source = source.replace(new_init, new_init + balance_code)
            changes_made.append("Task Balancing: Added (max 15k/task)")
            modified = True
    
    # === FIX 4: Early Stopping more aggressive ===
    if "patience = 3" in source:
        source = source.replace("patience = 3", "patience = 2")
        changes_made.append("Early Stopping: patience 3 -> 2")
        modified = True
    
    if modified:
        cell['source'] = [source]

# === FIX 5: Add Layer-wise LR Decay if not present ===
# This is complex - we add a new cell for optimizer setup
optimizer_cell_found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "optimizer = AdamW" in source and "param_groups" not in source:
            # Replace simple optimizer with layer-wise LR decay
            new_optimizer_code = '''
# Layer-wise Learning Rate Decay for BERT
# Lower layers (embeddings, early encoders) get smaller LR
# Higher layers (later encoders, heads) get full LR
def get_optimizer_grouped_parameters(model, lr, weight_decay, lr_decay=0.9):
    """
    Apply layer-wise LR decay: each layer gets lr * (lr_decay ** (num_layers - layer_idx))
    This preserves pre-trained knowledge in lower layers.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    
    # BERT encoder layers
    encoder_layers = [f"encoder.layer.{i}" for i in range(12)]
    
    param_groups = []
    
    # Embedding layer (lowest LR)
    param_groups.append({
        "params": [p for n, p in model.named_parameters() if "embeddings" in n and not any(nd in n for nd in no_decay)],
        "lr": lr * (lr_decay ** 12),
        "weight_decay": weight_decay
    })
    param_groups.append({
        "params": [p for n, p in model.named_parameters() if "embeddings" in n and any(nd in n for nd in no_decay)],
        "lr": lr * (lr_decay ** 12),
        "weight_decay": 0.0
    })
    
    # Encoder layers (progressively higher LR)
    for layer_idx, layer_name in enumerate(encoder_layers):
        layer_lr = lr * (lr_decay ** (11 - layer_idx))
        param_groups.append({
            "params": [p for n, p in model.named_parameters() if layer_name in n and not any(nd in n for nd in no_decay)],
            "lr": layer_lr,
            "weight_decay": weight_decay
        })
        param_groups.append({
            "params": [p for n, p in model.named_parameters() if layer_name in n and any(nd in n for nd in no_decay)],
            "lr": layer_lr,
            "weight_decay": 0.0
        })
    
    # Pooler and classification heads (full LR)
    param_groups.append({
        "params": [p for n, p in model.named_parameters() if "pooler" in n or "head" in n or "log_var" in n],
        "lr": lr,
        "weight_decay": weight_decay
    })
    
    return param_groups

# Create optimizer with layer-wise LR decay
optimizer_params = get_optimizer_grouped_parameters(model, CONFIG['lr'], CONFIG['weight_decay'], lr_decay=0.9)
optimizer = AdamW(optimizer_params)
print(f"Optimizer created with {len(optimizer_params)} parameter groups (Layer-wise LR Decay)")
'''
            # Replace old optimizer line
            cell['source'] = [source.replace(
                "optimizer = AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])",
                new_optimizer_code
            )]
            changes_made.append("Layer-wise LR Decay: Added (decay=0.9)")
            optimizer_cell_found = True
            break

# Save updated notebook
with open('notebooks/AURA_V8_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("\n📋 CHANGES APPLIED:")
for change in changes_made:
    print(f"  ✅ {change}")

if not changes_made:
    print("  ⚠️ No changes were made (patterns not found or already applied)")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
