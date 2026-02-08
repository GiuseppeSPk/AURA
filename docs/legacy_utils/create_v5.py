import json

# Read V4 notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        if 'AURA V4' in source:
            cell['source'] = [
                "# AURA V5: Optimized Hyperparameters\n",
                "\n",
                "---\n",
                "## PRIMA DI ESEGUIRE:\n",
                "1. **Settings** -> **Accelerator** -> **GPU T4 x2**\n",
                "2. **Add Input** -> Carica `aura-data-v2`\n",
                "---\n",
                "\n",
                "### V5 Optimizations\n",
                "| Parameter | V4 | V5 | Rationale |\n",
                "|-----------|----|----|----------|\n",
                "| Learning Rate | 2e-5 | **1e-5** | Slower convergence |\n",
                "| Dropout | 0.1 | **0.3** | More regularization |\n",
                "| Weight Decay | 0.01 | **0.02** | Stronger L2 |\n",
                "| Epochs | 5 | **8** | More time to converge |\n",
                "| Patience | 2 | **3** | More tolerance |\n",
                "\n",
                "**Goal**: Reduce overfitting gap while maintaining or improving F1."
            ]
            break

# Update CONFIG
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "CONFIG = {" in source and "'lr': 2e-5" in source:
            new_source = source.replace("'lr': 2e-5", "'lr': 1e-5")
            new_source = new_source.replace("'dropout': 0.1", "'dropout': 0.3")
            new_source = new_source.replace("'weight_decay': 0.01", "'weight_decay': 0.02")
            new_source = new_source.replace("'epochs': 5", "'epochs': 8")
            new_source = new_source.replace("'patience': 2", "'patience': 3")
            new_source = new_source.replace("V4: Focal Loss enabled", "V5: Optimized Hyperparameters")
            cell['source'] = [new_source]
            break

# Update training loop print
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'STARTING V4 TRAINING' in source:
            new_source = source.replace(
                'STARTING V4 TRAINING (Clean Data + Focal Loss)',
                'STARTING V5 TRAINING (Optimized: LR=1e-5, Dropout=0.3)'
            )
            new_source = new_source.replace('aura_v4_focal_best.pt', 'aura_v5_optimized_best.pt')
            cell['source'] = [new_source]

# Update final evaluation
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'V4 - Clean Data + Focal Loss' in source:
            new_source = source.replace(
                'V4 - Clean Data + Focal Loss',
                'V5 - Optimized Hyperparameters'
            )
            new_source = new_source.replace('aura_v4_focal_best.pt', 'aura_v5_optimized_best.pt')
            cell['source'] = [new_source]

# Save V5 notebook
with open('notebooks/AURA_Bayesian_V5_Optimized.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("=" * 50)
print("V5 NOTEBOOK CREATED")
print("=" * 50)
print("Optimizations applied:")
print("  - LR: 2e-5 -> 1e-5")
print("  - Dropout: 0.1 -> 0.3")
print("  - Weight Decay: 0.01 -> 0.02")
print("  - Epochs: 5 -> 8")
print("  - Patience: 2 -> 3")
print("=" * 50)
print("File: notebooks/AURA_Bayesian_V5_Optimized.ipynb")
