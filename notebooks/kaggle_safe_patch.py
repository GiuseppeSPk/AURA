import nbformat

def apply_kaggle_safe_logging(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found = False
    for cell in nb.cells:
        if 'def train_epoch(epoch):' in cell.source:
            old_source = cell.source
            
            # 1. Update tqdm mininterval to 10 seconds to reduce message frequency
            new_source = old_source.replace(
                "pbar = tqdm(train_loader, desc=f'Epoch {epoch}')",
                "pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=10.0)"
            )
            
            # 2. Limit pbar.set_postfix updates to every 50 steps
            # We look for the line: pbar.set_postfix({'loss': f'{loss.item() * CONFIG["gradient_accumulation"]:.3f}'})
            # And wrap it in an if condition
            old_postfix = "pbar.set_postfix({'loss': f'{loss.item() * CONFIG[\"gradient_accumulation\"]:.3f}'})"
            new_postfix = "if step % 50 == 0: pbar.set_postfix({'loss': f'{loss.item() * CONFIG[\"gradient_accumulation\"]:.3f}'})"
            
            new_source = new_source.replace(old_postfix, new_postfix)
            
            if new_source != old_source:
                cell.source = new_source
                found = True
                print(f"Applied Kaggle-safe logging to {notebook_path}")
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        print("Training loop NOT found or already patched.")

if __name__ == "__main__":
    apply_kaggle_safe_logging('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb')
