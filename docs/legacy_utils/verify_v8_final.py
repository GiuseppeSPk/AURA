import json

print('VALIDATING NOTEBOOK SYNTAX...')

with open('notebooks/AURA_V8_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

errors = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        code = ''.join(cell['source'])
        try:
            compile(code, f'cell_{i}', 'exec')
        except SyntaxError as e:
            errors.append(f'Cell {i}: {e}')

if errors:
    print('SYNTAX ERRORS FOUND:')
    for err in errors:
        print(f'  {err}')
else:
    print('ALL CELLS PASS SYNTAX CHECK')

# Verify key configs
full_code = ''
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        full_code += ''.join(cell['source'])

print('\nCONFIG VERIFICATION:')

lr_check = "'lr': 5e-6" in full_code
print(f"  LR 5e-6: {'YES' if lr_check else 'NO'}")

dropout_check = "'dropout': 0.4" in full_code
print(f"  Dropout 0.4: {'YES' if dropout_check else 'NO'}")

wd_check = "'weight_decay': 0.03" in full_code
print(f"  Weight Decay 0.03: {'YES' if wd_check else 'NO'}")

ep_check = "'epochs': 4" in full_code
print(f"  Epochs 4: {'YES' if ep_check else 'NO'}")

es_check = "no_improve_count" in full_code
print(f"  Early Stopping: {'YES' if es_check else 'NO'}")

bal_check = "balance_dataset" in full_code
print(f"  Task Balancing: {'YES' if bal_check else 'NO'}")

ls_check = "label_smoothing" in full_code
print(f"  Label Smoothing: {'YES' if ls_check else 'NO'}")
