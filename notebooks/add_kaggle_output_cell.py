import nbformat

# Load notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

# Create final output cell
final_cell = nbformat.v4.new_code_cell("""# Cell FINAL: Confirm Outputs Saved

import os
import json

print('='*60)
print('📦 TRAINING COMPLETE - OUTPUT VERIFICATION')
print('='*60)

# Check model file
if os.path.exists('aura_v10_best.pt'):
    size_mb = os.path.getsize('aura_v10_best.pt') / 1e6
    print(f'✅ Model saved: aura_v10_best.pt ({size_mb:.1f} MB)')
else:
    print('❌ WARNING: Model file not found!')

# Check history file
if os.path.exists('aura_v10_history.json'):
    with open('aura_v10_history.json', 'r') as f:
        history = json.load(f)
    print(f'✅ History saved: {len(history["val_f1"])} epochs recorded')
    print(f'   Best Val F1: {history["best_f1"]:.4f}')
else:
    print('❌ WARNING: History file not found!')

print('\\n' + '='*60)
print('🎉 All outputs saved to /kaggle/working/')
print('   Download from the OUTPUT tab (right panel) →')
print('='*60)
""")

# Add to notebook
nb.cells.append(final_cell)

# Save
nbformat.write(nb, 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb')

print("✅ Added final output verification cell to notebook")
print("\n📋 NEXT STEPS:")
print("1. Upload AURA_V10_PROD.ipynb to Kaggle")
print("2. Attach dataset: aura-v10-data")
print("3. Click 'Save Version' → 'Save & Run All (Commit)'")
print("4. Select GPU: T4 x2 or P100")
print("5. Close browser and wait for email notification!")
print("\n⏰ Training time: ~8.8 hours")
print("📧 Kaggle will email you when complete")
