import nbformat

def update_config_for_accuracy(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found = False
    for cell in nb.cells:
        if "CONFIG = {" in cell.source and "'epochs': 5" in cell.source:
            old_source = cell.source
            new_source = old_source.replace("'epochs': 5", "'epochs': 12")
            new_source = new_source.replace("'patience': 3", "'patience': 5")
            new_source = new_source.replace("'lr_encoder': 2e-5", "'lr_encoder': 1e-5")
            
            if new_source != old_source:
                cell.source = new_source
                found = True
                print(f"Updated CONFIG for accuracy in {notebook_path}")
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        print("CONFIG NOT found or already updated.")

if __name__ == "__main__":
    update_config_for_accuracy('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb')
