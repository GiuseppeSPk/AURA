import json

# Update V4 notebook to make gamma configurable
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find CONFIG cell and add focal_gamma
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'mc_samples': 10" in source and "CONFIG = {" in source:
            # Add focal_gamma parameter
            new_source = source.replace(
                "'mc_samples': 10,\n    'output_dir': '/kaggle/working'\n}",
                "'mc_samples': 10,\n    'focal_gamma': 2.0,  # Ablation: Test [1.0, 2.0, 3.0]\n    'output_dir': '/kaggle/working'\n}"
            )
            cell['source'] = [new_source]
            break

# Update focal_loss function call to use config
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'focal_loss_with_uncertainty(' in source and 'gamma=2.0' in source:
            new_source = source.replace(
                'gamma=2.0,',
                "gamma=config['focal_gamma'],"
            )
            cell['source'] = [new_source]

# Save updated notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("✅ V4 Updated: focal_gamma is now configurable")
print("   To test different values, change: CONFIG['focal_gamma'] = X.X")
print("   Suggested ablation: [1.0, 2.0, 3.0]")
