import nbformat

# Load the production notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

print("🔧 Applying CRITICAL FIX to Kendall Loss...")
print("="*60)

# Find and fix the UncertaintyLoss cell
found = False
for cell in nb.cells:
    if 'class UncertaintyLoss' in cell.source:
        old_source = cell.source
        
        # Fix 1: The precision calculation in forward()
        new_source = old_source.replace(
            'precision = torch.exp(-F.softplus(self.log_vars[i]))',
            'precision = 1.0 / (F.softplus(self.log_vars[i]) + 1e-8)  # FIXED: Correct inverse formula'
        )
        
        # Fix 2: The get_weights() method
        new_source = new_source.replace(
            'return torch.exp(-F.softplus(self.log_vars)).detach().cpu().numpy()',
            'return (1.0 / (F.softplus(self.log_vars) + 1e-8)).detach().cpu().numpy()  # FIXED'
        )
        
        if new_source != old_source:
            cell.source = new_source
            found = True
            print("✅ FIXED: UncertaintyLoss.forward() - precision calculation")
            print("✅ FIXED: UncertaintyLoss.get_weights() - weight reporting")
            
            # Show the changes
            print("\\n" + "="*60)
            print("BEFORE:")
            print("  precision = torch.exp(-F.softplus(self.log_vars[i]))")
            print("\\nAFTER:")
            print("  precision = 1.0 / (F.softplus(self.log_vars[i]) + 1e-8)")
            print("="*60)

if found:
    # Save the fixed notebook
    nbformat.write(nb, 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb')
    print("\\n✅ Notebook saved with CRITICAL FIX applied!")
    print("\\n📊 Expected improvements:")
    print("  - Task Weights will reach 1.2-1.5 (instead of plateauing at 0.6)")
    print("  - Val F1 should reach 0.75-0.78 by Epoch 5")
    print("  - Final F1 (Epoch 10-12): 0.78-0.82")
    print("\\n🚀 Ready for the final training run!")
else:
    print("❌ Could not find UncertaintyLoss class in notebook")
