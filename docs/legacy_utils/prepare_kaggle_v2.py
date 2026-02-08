# Final Dataset Preparation for Kaggle V2
import pandas as pd
import shutil
from pathlib import Path

print("="*60)
print("PREPARING AURA-DATA-V2 FOR KAGGLE")
print("="*60)

SRC = Path("data/processed")
DST = Path("data/kaggle_upload_v2")
DST.mkdir(exist_ok=True)

# Files to include
files_to_copy = {
    'olid_train.csv': 'olid_train.csv',
    'olid_validation_clean.csv': 'olid_validation.csv',  # Rename for consistency
    'goemotions_clean.csv': 'goemotions_clean.csv'
}

print("\n1. COPYING FILES...")
for src_name, dst_name in files_to_copy.items():
    src_path = SRC / src_name
    dst_path = DST / dst_name
    if src_path.exists():
        shutil.copy(src_path, dst_path)
        print(f"   {src_name} -> {dst_name}")
    else:
        print(f"   ERROR: {src_name} not found!")

print("\n2. FINAL VALIDATION...")
print("-"*60)

# Load and validate each file
all_ok = True

# OLID Train
olid_train = pd.read_csv(DST / 'olid_train.csv')
print(f"\nolid_train.csv:")
print(f"   Rows: {len(olid_train)}")
print(f"   Columns: {list(olid_train.columns)}")
print(f"   Missing text: {olid_train['text'].isna().sum()}")
print(f"   Label distribution: NOT={sum(olid_train['label']==0)}, OFF={sum(olid_train['label']==1)}")

# OLID Validation
olid_val = pd.read_csv(DST / 'olid_validation.csv')
print(f"\nolid_validation.csv:")
print(f"   Rows: {len(olid_val)}")
print(f"   Missing text: {olid_val['text'].isna().sum()}")

# Check overlap
train_texts = set(olid_train['text'].str.lower().str.strip())
val_texts = set(olid_val['text'].str.lower().str.strip())
overlap = len(train_texts & val_texts)
print(f"   Train/Val overlap: {overlap}")
if overlap > 0:
    print("   ERROR: Overlap detected!")
    all_ok = False

# GoEmotions
goemo = pd.read_csv(DST / 'goemotions_clean.csv')
emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
print(f"\ngoemotions_clean.csv:")
print(f"   Rows: {len(goemo)}")
print(f"   Missing text: {goemo['text'].isna().sum()}")
print(f"   Emotion distribution:")
for col in emo_cols:
    count = goemo[col].sum()
    print(f"      {col}: {count}")
    if count == 0:
        print(f"      ERROR: {col} has 0 samples!")
        all_ok = False

# All-zero check
all_zero = (goemo[emo_cols].sum(axis=1) == 0).sum()
print(f"   All-zero rows: {all_zero}")
if all_zero > 0:
    all_ok = False

print("\n" + "="*60)
if all_ok:
    print("ALL CHECKS PASSED!")
    print(f"\nUpload folder: {DST.absolute()}")
    print("\nFiles ready for Kaggle:")
    for f in DST.glob("*.csv"):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name} ({size_mb:.2f} MB)")
else:
    print("ERRORS FOUND! Do not upload.")
print("="*60)
