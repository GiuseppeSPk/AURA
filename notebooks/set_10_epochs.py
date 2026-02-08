import nbformat

# Load the production notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

print("🔧 Updating configuration for final run...")
print("="*60)

# Find and update the CONFIG cell
found = False
for cell in nb.cells:
    if "'epochs': 12" in cell.source:
        old_source = cell.source
        
        # Change epochs from 12 to 10
        new_source = old_source.replace("'epochs': 12,", "'epochs': 10,  # FINAL RUN: Optimized for Kaggle timeout")
        
        if new_source != old_source:
            cell.source = new_source
            found = True
            print("✅ Updated: epochs 12 → 10")
            print("   Estimated training time: ~8.5 hours (safe for Kaggle)")

if found:
    # Save the updated notebook
    nbformat.write(nb, 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb')
    print("\n✅ Notebook updated and saved!")
    print("\n📊 Expected results (10 epochs with Kendall fix):")
    print("  - Epoch 5 Val F1:  0.75-0.78")
    print("  - Epoch 10 Val F1: 0.78-0.80")
    print("  - Task Weights:    1.25-1.45 (well-differentiated)")
    print("\n🚀 Ready to upload to Kaggle!")
else:
    print("⚠️ Could not find epochs configuration in notebook")
