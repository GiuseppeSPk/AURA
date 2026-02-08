import nbformat

# Load the production notebook
nb = nbformat.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_PROD.ipynb', as_version=4)

print("🔧 Creating COLAB version of AURA V10...")
print("="*60)

# Create new cells for Colab setup
colab_setup_cells = []

# Cell 1: GPU Check
gpu_check = nbformat.v4.new_code_cell("""# Colab Setup: GPU Check
import torch
print("🔧 Checking GPU availability...")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ WARNING: No GPU detected!")
    print("   Go to Runtime → Change runtime type → GPU (T4)")
""")

# Cell 2: Mount Google Drive
drive_mount = nbformat.v4.new_code_cell("""# Colab Setup: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")
print("\\n📂 Upload the 'aura-v10-data' folder to: My Drive/")
""")

# Cell 3: Install dependencies
install_deps = nbformat.v4.new_code_cell("""# Colab Setup: Install Dependencies
!pip install -q transformers datasets
print("✅ Dependencies installed!")
""")

# Add setup cells at the beginning (after first markdown cell)
colab_nb_cells = [nb.cells[0]]  # Keep intro markdown
colab_nb_cells.extend([gpu_check, drive_mount, install_deps])

# Modify CONFIG cell to use Google Drive path
for cell in nb.cells[1:]:
    if "'encoder': 'roberta-base'" in cell.source:
        # This is the CONFIG cell - update the DATA_DIR
        old_source = cell.source
        new_source = old_source.replace(
            "DATA_DIR = '/kaggle/input/aura-v10-data'",
            "# COLAB: Update this path if your folder is elsewhere\nDATA_DIR = '/content/drive/MyDrive/aura-v10-data'"
        )
        # Also reduce epochs for testing
        new_source = new_source.replace(
            "'epochs': 10,  # FINAL RUN",
            "'epochs': 3,  # COLAB TEST: Reduced for quick validation (change to 10 for full run)"
        )
        cell.source = new_source
        print("✅ Modified CONFIG: Google Drive path + 3 epochs for testing")
    
    colab_nb_cells.append(cell)

# Create new notebook
colab_nb = nbformat.v4.new_notebook()
colab_nb.cells = colab_nb_cells
colab_nb.metadata = nb.metadata

# Save
output_path = 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_COLAB.ipynb'
nbformat.write(colab_nb, output_path)

print(f"\n✅ Created COLAB notebook!")
print(f"   Original cells: {len(nb.cells)}")
print(f"   Colab cells: {len(colab_nb_cells)} (added 3 setup cells)")
print(f"\n📂 Saved to: AURA_V10_COLAB.ipynb")
print(f"\n🚀 NEXT STEPS:")
print(f"   1. Upload 'aura-v10-data' folder to Google Drive")
print(f"   2. Upload AURA_V10_COLAB.ipynb to Colab")
print(f"   3. Runtime → Change runtime type → T4 GPU")
print(f"   4. Run all cells!")
print(f"\n⏰ Test run (3 epochs): ~2.5 hours")
print(f"   Full run (10 epochs): ~8.5 hours")
