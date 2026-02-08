import json

print('='*70)
print('AURA V8 DEEP CODE AUDIT')
print('='*70)

with open('notebooks/AURA_V8_Colab.ipynb', 'r') as f:
    nb = json.load(f)

errors = []
warnings = []

# Extract all code
all_code = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        all_code.append((i, source))

full_code = '\n'.join([c[1] for c in all_code])

# CHECK 1: Custom collate exists
print('\n[1] CUSTOM COLLATE')
if 'def custom_collate' in full_code:
    print('  [OK] custom_collate defined')
    if 'collate_fn=custom_collate' in full_code:
        print('  [OK] collate_fn used in DataLoader')
    else:
        errors.append('collate_fn not passed to DataLoader')
        print('  [FAIL] collate_fn not used')
else:
    errors.append('custom_collate not defined')
    print('  [FAIL] custom_collate missing')

# CHECK 2: Label extraction for binary tasks
print('\n[2] LABEL EXTRACTION')
if 'labels[mask][:, 0].long()' in full_code:
    print('  [OK] Binary task labels extracted correctly')
else:
    errors.append('Binary label extraction missing')
    print('  [FAIL] Need labels[mask][:, 0].long() for binary tasks')

# CHECK 3: All 4 tasks processed
print('\n[3] TASK PROCESSING')
for task in ['toxicity', 'sentiment', 'hate', 'emotion']:
    if f"'{task}'" in full_code or f'"{task}"' in full_code:
        print(f'  [OK] {task} task processed')
    else:
        warnings.append(f'{task} task not found')
        print(f'  [WARN] {task} task not found')

# CHECK 4: Model heads
print('\n[4] MODEL ARCHITECTURE')
heads = ['toxicity_head', 'emotion_head', 'sentiment_head', 'hate_head']
for head in heads:
    if head in full_code:
        print(f'  [OK] {head} defined')
    else:
        errors.append(f'{head} missing')
        print(f'  [FAIL] {head} missing')

# CHECK 5: Log vars
print('\n[5] KENDALL LOG VARS')
log_vars = ['tox_log_var', 'emo_log_var', 'sent_log_var', 'hate_log_var']
for lv in log_vars:
    if lv in full_code:
        print(f'  [OK] {lv} defined')
    else:
        errors.append(f'{lv} missing')
        print(f'  [FAIL] {lv} missing')

# CHECK 6: Loss functions
print('\n[6] LOSS FUNCTIONS')
if 'def focal_loss_with_uncertainty' in full_code:
    print('  [OK] focal_loss_with_uncertainty defined')
else:
    errors.append('focal_loss_with_uncertainty missing')
    print('  [FAIL] focal_loss_with_uncertainty missing')

if 'def mc_bce_loss' in full_code:
    print('  [OK] mc_bce_loss defined')
else:
    errors.append('mc_bce_loss missing')
    print('  [FAIL] mc_bce_loss missing')

# CHECK 7: Data paths
print('\n[7] DATA PATHS')
if 'aura_v8_data.zip' in full_code:
    print('  [OK] V8 zip name correct')
else:
    warnings.append('V8 zip name not found')
    print('  [WARN] aura_v8_data.zip not found')

if 'aura_v8_best.pt' in full_code:
    print('  [OK] V8 model save name correct')
else:
    warnings.append('V8 model name not found')
    print('  [WARN] aura_v8_best.pt not found')

# CHECK 8: Imports
print('\n[8] IMPORTS')
required_imports = ['torch', 'transformers', 'pandas', 'sklearn']
for imp in required_imports:
    if imp in full_code:
        print(f'  [OK] {imp} imported')
    else:
        errors.append(f'{imp} not imported')
        print(f'  [FAIL] {imp} not imported')

# CHECK 9: GPU setup
print('\n[9] GPU SETUP')
if 'torch.cuda.is_available()' in full_code:
    print('  [OK] CUDA check present')
else:
    warnings.append('CUDA check missing')
    print('  [WARN] CUDA check missing')

# CHECK 10: Gradient clipping
print('\n[10] TRAINING SAFETY')
if 'clip_grad_norm_' in full_code:
    print('  [OK] Gradient clipping present')
else:
    warnings.append('Gradient clipping missing')
    print('  [WARN] Gradient clipping missing')

if 'optimizer.zero_grad()' in full_code:
    print('  [OK] Gradient zeroing present')
else:
    errors.append('optimizer.zero_grad() missing')
    print('  [FAIL] optimizer.zero_grad() missing')

# SUMMARY
print('\n' + '='*70)
if errors:
    print(f'ERRORS ({len(errors)}):')
    for e in errors:
        print(f'  - {e}')
    print('\nSTATUS: FIX REQUIRED')
else:
    print('STATUS: ALL CHECKS PASSED')
    if warnings:
        print(f'\nWarnings ({len(warnings)}):')
        for w in warnings:
            print(f'  - {w}')
print('='*70)
