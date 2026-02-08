"""
AURA Pro Simple: Dataset Curation Script
=========================================
Creates a unified binary dataset for DeBERTa-v3-base fine-tuning.

Logic:
- Toxicity samples: Keep original labels (1=toxic, 0=non-toxic)
- Reporting samples (is_reporting=1): → label=0 (safe, it's just reporting)
- Reporting samples (is_reporting=0): → label=1 (toxic, direct statement)

This teaches the model that REPORTING toxic content is NOT toxic itself.
"""

import pandas as pd
import numpy as np

SEED = 42
np.random.seed(SEED)

DATA_DIR = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data'

def create_unified_dataset():
    print("🔧 Creating Unified Toxicity + Reporting Dataset...")
    
    # 1. Load Toxicity (binary: 0=non-toxic, 1=toxic)
    tox_df = pd.read_csv(f'{DATA_DIR}/toxicity_train.csv')
    tox_df = tox_df[['text', 'label']].copy()
    tox_df['source'] = 'toxicity'
    print(f"   ✓ Toxicity: {len(tox_df):,} samples")
    print(f"      Distribution: {tox_df['label'].value_counts().to_dict()}")
    
    # 2. Load Reporting (is_reporting: 0=direct, 1=reporting)
    rep_df = pd.read_csv(f'{DATA_DIR}/reporting_examples_augmented.csv')
    
    # Map reporting labels to toxicity labels:
    # - is_reporting=1 → These are "safe" (reporting context) → label=0
    # - is_reporting=0 → These are direct toxic statements → label=1
    rep_df['label'] = rep_df['is_reporting'].apply(lambda x: 0 if x == 1 else 1)
    rep_df = rep_df[['text', 'label']].copy()
    rep_df['source'] = 'reporting'
    print(f"   ✓ Reporting: {len(rep_df):,} samples")
    print(f"      Distribution after mapping: {rep_df['label'].value_counts().to_dict()}")
    
    # 3. Combine
    unified_df = pd.concat([tox_df, rep_df], ignore_index=True)
    
    # 4. Shuffle
    unified_df = unified_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # 5. Save
    output_path = f'{DATA_DIR}/deberta_unified_train.csv'
    unified_df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("📊 UNIFIED DATASET SUMMARY")
    print("="*60)
    print(f"Total samples: {len(unified_df):,}")
    print(f"Label distribution:")
    print(f"   0 (Non-Toxic / Reporting): {(unified_df['label'] == 0).sum():,}")
    print(f"   1 (Toxic Direct):          {(unified_df['label'] == 1).sum():,}")
    print(f"\nSource breakdown:")
    print(f"   Toxicity: {(unified_df['source'] == 'toxicity').sum():,}")
    print(f"   Reporting: {(unified_df['source'] == 'reporting').sum():,}")
    print(f"\n✅ Saved to: {output_path}")
    
    return unified_df

def create_validation_set():
    """Create validation set from toxicity_val + reporting_validation_clean"""
    print("\n🔧 Creating Unified Validation Dataset...")
    
    # Toxicity validation
    tox_val = pd.read_csv(f'{DATA_DIR}/toxicity_val.csv')
    tox_val = tox_val[['text', 'label']].copy()
    tox_val['source'] = 'toxicity'
    
    # Reporting validation
    rep_val = pd.read_csv(f'{DATA_DIR}/reporting_validation_clean.csv')
    rep_val['label'] = rep_val['is_reporting'].apply(lambda x: 0 if x == 1 else 1)
    rep_val = rep_val[['text', 'label']].copy()
    rep_val['source'] = 'reporting'
    
    # Combine
    val_df = pd.concat([tox_val, rep_val], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    output_path = f'{DATA_DIR}/deberta_unified_val.csv'
    val_df.to_csv(output_path, index=False)
    
    print(f"   ✓ Validation samples: {len(val_df):,}")
    print(f"   ✅ Saved to: {output_path}")
    
    return val_df

if __name__ == "__main__":
    train_df = create_unified_dataset()
    val_df = create_validation_set()
    
    print("\n" + "="*60)
    print("🎯 SAMPLE VERIFICATION")
    print("="*60)
    
    # Show some examples
    print("\nReporting samples (should be label=0):")
    rep_samples = train_df[train_df['source'] == 'reporting'].head(3)
    for _, row in rep_samples.iterrows():
        print(f"   [{row['label']}] {row['text'][:60]}...")
    
    print("\nToxic samples (should be label=1):")
    tox_samples = train_df[(train_df['source'] == 'toxicity') & (train_df['label'] == 1)].head(3)
    for _, row in tox_samples.iterrows():
        print(f"   [{row['label']}] {row['text'][:60]}...")
