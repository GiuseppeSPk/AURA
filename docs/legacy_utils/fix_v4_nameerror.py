import json

# Read V4 notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix function definition
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "def focal_loss_with_uncertainty(logits, log_var, targets, gamma=config['focal_gamma']" in source:
            # Replace with a safe default (2.0)
            new_source = source.replace(
                "def focal_loss_with_uncertainty(logits, log_var, targets, gamma=config['focal_gamma'], T=10):",
                "def focal_loss_with_uncertainty(logits, log_var, targets, gamma=2.0, T=10):"
            )
            cell['source'] = [new_source]
            print("Fixed function definition parameter.")

# Save fixed V4 notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("✅ V4 Notebook NameError fixed!")
