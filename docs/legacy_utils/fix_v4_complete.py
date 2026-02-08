import json

# Read V4 notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes_applied = []

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        new_source = source
        
        # Fix 1: Remove class_weights from train_epoch function signature
        if 'def train_epoch(model, loader, optimizer, scheduler, epoch, config, class_weights):' in source:
            new_source = new_source.replace(
                'def train_epoch(model, loader, optimizer, scheduler, epoch, config, class_weights):',
                'def train_epoch(model, loader, optimizer, scheduler, epoch, config):'
            )
            fixes_applied.append("Fixed train_epoch signature (removed class_weights)")
        
        # Fix 2: Remove CLASS_WEIGHTS from training loop call
        if 'train_epoch(model, train_loader, optimizer, scheduler, epoch, CONFIG, CLASS_WEIGHTS)' in source:
            new_source = new_source.replace(
                'train_epoch(model, train_loader, optimizer, scheduler, epoch, CONFIG, CLASS_WEIGHTS)',
                'train_epoch(model, train_loader, optimizer, scheduler, epoch, CONFIG)'
            )
            fixes_applied.append("Fixed train_epoch call (removed CLASS_WEIGHTS)")
        
        # Fix 3: Correct the model file path in final evaluation
        if 'aura_v3_best.pt' in source:
            new_source = new_source.replace('aura_v3_best.pt', 'aura_v4_focal_best.pt')
            fixes_applied.append("Fixed model path: v3 -> v4")
        
        # Fix 4: Correct the print statement
        if 'V3 - Clean Data + Class Weights' in source:
            new_source = new_source.replace(
                'V3 - Clean Data + Class Weights',
                'V4 - Clean Data + Focal Loss'
            )
            fixes_applied.append("Fixed print label: V3 -> V4")
        
        if new_source != source:
            cell['source'] = [new_source]
    
    # Fix 5: Update markdown header
    elif cell['cell_type'] == 'markdown':
        source_list = cell['source']
        for i, line in enumerate(source_list):
            if 'with Class Weights' in line:
                source_list[i] = line.replace('with Class Weights', 'with Focal Loss')
                fixes_applied.append("Fixed markdown header")

# Save fixed notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("=" * 50)
print("V4 NOTEBOOK COMPREHENSIVE FIX")
print("=" * 50)
for fix in fixes_applied:
    print(f"  [OK] {fix}")
print("=" * 50)
print(f"Total fixes: {len(fixes_applied)}")
print("Notebook ready for Kaggle upload!")
