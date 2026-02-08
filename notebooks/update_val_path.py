import json
import os

path = r"C:\Users\spicc\Desktop\Multimodal\AURA\notebooks\AURA_V10_COLAB.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the data loading cell
for cell in nb['cells']:
    src = "".join(cell['source'])
    if "reporting_validation.csv" in src and "rep_val = " in src:
        print("Found data loading cell, updating...")
        new_src = src.replace("reporting_validation.csv", "reporting_validation_clean.csv")
        cell['source'] = [new_src]
        break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebook to use reporting_validation_clean.csv")
