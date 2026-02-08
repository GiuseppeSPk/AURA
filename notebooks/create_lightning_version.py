import nbformat

# Load the production notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

print("🔧 Creating LIGHTNING.AI version...")
print("="*60)

# Modify CONFIG cell for Lightning.ai paths
for cell in nb.cells:
    if "'encoder': 'roberta-base'" in cell.source:
        old_source = cell.source
        # Change data directory path
        new_source = old_source.replace(
            "DATA_DIR = '/kaggle/input/aura-v10-data'",
            "# Lightning.ai path (upload dataset to studio root)\nDATA_DIR = './aura-v10-data'"
        )
        # Boost batch size for L40S (48GB VRAM!)
        new_source = new_source.replace(
            "'batch_size': 16,",
            "'batch_size': 32,  # L40S has 48GB VRAM, can handle larger batches!"
        )
        cell.source = new_source
        print("✅ Modified CONFIG:")
        print("   - Data path: ./aura-v10-data")
        print("   - Batch size: 32 (2x faster!)")

# Save
output_path = 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_LIGHTNING.ipynb'
nbformat.write(nb, output_path)

print(f"\n✅ Lightning.ai notebook ready!")
print(f"   Saved to: AURA_V10_LIGHTNING.ipynb")
print(f"\n📂 Total files to upload: 6")
print(f"   - 1 notebook (AURA_V10_LIGHTNING.ipynb)")
print(f"   - 5 CSV files (from aura-v10-data folder)")
