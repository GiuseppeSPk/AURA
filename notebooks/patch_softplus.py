import nbformat

def patch_softplus(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found = False
    for cell in nb.cells:
        if 'class UncertaintyLoss' in cell.source:
            # Update forward and get_weights to use SoftPlus
            old_source = cell.source
            new_source = old_source.replace(
                "precision = torch.exp(-self.log_vars[i])",
                "# SoftPlus variant for better numerical stability\n            precision = torch.exp(-F.softplus(self.log_vars[i]))"
            ).replace(
                "total += precision * loss + self.log_vars[i] * 0.5",
                "total += precision * loss + F.softplus(self.log_vars[i]) * 0.5"
            ).replace(
                "return torch.exp(-self.log_vars).detach().cpu().numpy()",
                "return torch.exp(-F.softplus(self.log_vars)).detach().cpu().numpy()"
            )
            
            if new_source != old_source:
                cell.source = new_source
                found = True
                print(f"Patched UncertaintyLoss in {notebook_path}")
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
    else:
        print("UncertaintyLoss NOT found or already patched.")

if __name__ == "__main__":
    patch_softplus('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_Final_Fixed.ipynb')
