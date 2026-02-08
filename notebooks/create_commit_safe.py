import nbformat

# Load the production notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

print("🔧 Creating COMMIT-SAFE version (no plotting, minimal logs)...")
print("="*60)

# Create new notebook with only essential cells
safe_cells = []

for i, cell in enumerate(nb.cells):
    cell_source = cell.source if hasattr(cell, 'source') else ''
    
    # SKIP plotting cells
    if any(keyword in cell_source for keyword in [
        'plot_class_distribution',
        'plot_confusion_matrix',
        'plot_multilabel_confusion',
        'plot_training_history',
        'plt.subplots',
        'plt.show()',
        'sns.heatmap',
        'matplotlib',
        'fig, axes'
    ]):
        print(f"⏭️ Skipping plotting cell {i+1}")
        continue
    
    # SKIP data distribution analysis cell (Cell 9)
    if 'CLASS DISTRIBUTION ANALYSIS' in cell_source:
        print(f"⏭️ Skipping distribution analysis cell {i+1}")
        continue
    
    # Keep all other cells
    safe_cells.append(cell)

# Create new notebook
safe_nb = nbformat.v4.new_notebook()
safe_nb.cells = safe_cells
safe_nb.metadata = nb.metadata

# Save
output_path = 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_COMMIT_SAFE.ipynb'
nbformat.write(safe_nb, output_path)

print(f"\n✅ Created COMMIT-SAFE notebook")
print(f"   Original cells: {len(nb.cells)}")
print(f"   Safe cells: {len(safe_cells)}")
print(f"   Removed: {len(nb.cells) - len(safe_cells)} plotting/analysis cells")
print(f"\n📂 Saved to: AURA_V10_COMMIT_SAFE.ipynb")
print(f"\n🚀 Upload THIS version to Kaggle for guaranteed commit success!")
