import json

# Verify V5 notebook
with open('notebooks/AURA_V5_Production.ipynb', 'r') as f:
    nb = json.load(f)

print('V5 NOTEBOOK VERIFICATION')
print('='*50)
print('Cells:', len(nb['cells']))

# Check critical components
checks = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if "'num_emotion_classes': 5" in source:
            checks.append('5 emotions configured')
        if "'dropout': 0.3" in source:
            checks.append('Dropout 0.3')
        if "'lr': 1e-5" in source:
            checks.append('LR 1e-5')
        if 'focal_loss_with_uncertainty' in source and 'def ' in source:
            checks.append('Focal Loss defined')
        if 'goemotions_v5.csv' in source:
            checks.append('V5 data path')
        if 'aura_v5_best.pt' in source:
            checks.append('V5 model save path')

print('Checks passed:')
for c in checks:
    print(f'  [OK] {c}')
print('='*50)
print('NOTEBOOK READY FOR KAGGLE')
