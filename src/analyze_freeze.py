import json

nb = json.load(open('c:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_Kaggle.ipynb', 'r', encoding='utf-8'))

print(f'Total cells: {len(nb["cells"])}')

freeze_keywords = ['freeze', 'requires_grad', 'progressive', 'unfreeze']
freeze_cells = []

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', ''))
    if any(kw in source.lower() for kw in freeze_keywords):
        freeze_cells.append((i, cell))

print(f'\nFound {len(freeze_cells)} cells with freezing logic\n')

for idx, cell in freeze_cells:
    print(f'{"="*60}')
    print(f'CELL {idx} - TYPE: {cell["cell_type"]}')
    print(f'{"="*60}')
    source_text = ''.join(cell['source'])
    print(source_text[:1500])
    print('\n')
