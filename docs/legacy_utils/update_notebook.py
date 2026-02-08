import json

# Read notebook
with open('notebooks/AURA_Bayesian_V3_CleanData.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the cell with path detection
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'DATA_DIR = None' in source and 'goemotions_clean.csv' in source:
            # Update with the subfolder path
            old_path = "for path in ['/kaggle/input/aura-data-v2', '/kaggle/input/aura_data_v2', '/kaggle/input/aura-data', 'data/processed', 'data/kaggle_upload_v2']:"
            new_path = "for path in ['/kaggle/input/aura-data-v2/kaggle_upload_v2', '/kaggle/input/aura-data-v2', '/kaggle/input/aura_data_v2/kaggle_upload_v2', '/kaggle/input/aura_data_v2', '/kaggle/input/aura-data', 'data/processed', 'data/kaggle_upload_v2']:"
            new_source = source.replace(old_path, new_path)
            cell['source'] = [new_source]
            print('Updated with subfolder path!')
            break

# Save
with open('notebooks/AURA_Bayesian_V3_CleanData.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print('Notebook saved!')
