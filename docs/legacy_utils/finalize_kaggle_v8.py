import json

# Convert Colab notebook back to Kaggle format
# Main difference: Data paths
# Colab: data/toxicity_train.csv (unzipped)
# Kaggle: /kaggle/input/aura-mega-data/toxicity_train.csv

with open('notebooks/AURA_V8_Kaggle.ipynb', 'r') as f:
    nb = json.load(f)

# Update Data Paths
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Update path
        if 'DATA_DIR =' in source:
            new_source = source.replace("DATA_DIR = 'data'", "DATA_DIR = '/kaggle/input/aura-mega-data'")
            cell['source'] = [new_source]
            print('Updated DATA_DIR')
        
        # Remove Colab specific setup
        if '!pip install' in source or '!unzip' in source:
            cell['source'] = ["# Kaggle Environment already has transformers installed\n# dataset is mounted at /kaggle/input/aura-mega-data"]
            print('Cleared Colab setup cell')

with open('notebooks/AURA_V8_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)

print('Kaggle Notebook Ready')
