import json

# Update the notebook to reflect 500 reporting samples
path = r'C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_Final_Fixed.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    # Update documentation cell
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "### 📊 Dataset Update: Reporting Task" in source:
            source = source.replace("After**: 298 examples", "After**: 500 examples")
            source = source.replace("298 examples (3x increase)", "500 examples (5x increase)")
            source = source.replace("149 Direct / 149 Reporting", "250 Direct / 250 Reporting")
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'][-1] = '' # clean up

    # Update code comments
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "reporting_examples.csv" in source:
            source = source.replace("298 samples", "500 samples")
            cell['source'] = [line + '\n' for line in source.split('\n')]
            if cell['source'][-1] == '\n': cell['source'][-1] = ''

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated to 500 samples.")
