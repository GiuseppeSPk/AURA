"""
V5 Data Preparation Script
--------------------------
Fixes the core data issues:
1. Balances GoEmotions (undersample Joy/Neutral, oversample Disgust/Fear)
2. Filters short/noisy OLID texts
3. Validates final dataset quality
"""
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

print("="*60)
print("V5 DATA PREPARATION")
print("="*60)

# Create output directory
os.makedirs('data/kaggle_upload_v5', exist_ok=True)

# ===========================================
# 1. LOAD DATA
# ===========================================
print("\n[1/4] Loading data...")
goemo = pd.read_csv('data/kaggle_upload_v2/goemotions_clean.csv')
olid_train = pd.read_csv('data/kaggle_upload_v2/olid_train.csv')
olid_val = pd.read_csv('data/kaggle_upload_v2/olid_validation.csv')

print(f"  GoEmotions: {len(goemo)} samples")
print(f"  OLID Train: {len(olid_train)} samples")
print(f"  OLID Val: {len(olid_val)} samples")

# ===========================================
# 2. BALANCE GOEMOTIONS
# ===========================================
print("\n[2/4] Balancing GoEmotions...")

emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# Target counts - reduce dominant, boost rare
targets = {
    'joy': 8000,       # Was 21,733 -> undersample
    'neutral': 8000,   # Was 17,772 -> undersample
    'anger': None,     # Keep all 7,022
    'surprise': None,  # Keep all 6,668
    'sadness': None,   # Keep all 4,032
    'disgust': 3000,   # Was 1,013 -> oversample 3x
    'fear': 2800       # Was 929 -> oversample 3x
}

balanced_dfs = []
for emo, target in targets.items():
    if emo not in goemo.columns:
        continue
    emo_data = goemo[goemo[emo] == 1].copy()
    original_count = len(emo_data)
    
    if target is None:
        # Keep all
        balanced_dfs.append(emo_data)
        print(f"  {emo:10}: {original_count:5} -> {original_count:5} (kept)")
    elif target < original_count:
        # Undersample
        sampled = emo_data.sample(target, random_state=42)
        balanced_dfs.append(sampled)
        print(f"  {emo:10}: {original_count:5} -> {target:5} (undersampled)")
    else:
        # Oversample with replacement
        sampled = resample(emo_data, n_samples=target, replace=True, random_state=42)
        balanced_dfs.append(sampled)
        print(f"  {emo:10}: {original_count:5} -> {target:5} (oversampled)")

# Combine and deduplicate
goemo_balanced = pd.concat(balanced_dfs).drop_duplicates(subset=['text'])
print(f"\n  Final GoEmotions: {len(goemo_balanced)} samples")

# Verify new distribution
print("\n  New distribution:")
for emo in emo_cols:
    if emo in goemo_balanced.columns:
        count = goemo_balanced[emo].sum()
        pct = count / len(goemo_balanced) * 100
        bar = '#' * int(pct / 2)
        print(f"    {emo:10}: {int(count):5} ({pct:5.1f}%) {bar}")

# ===========================================
# 3. FILTER OLID SHORT TEXTS
# ===========================================
print("\n[3/4] Filtering short OLID texts...")

min_length = 25
olid_train_clean = olid_train[olid_train['text'].str.len() >= min_length].copy()
removed = len(olid_train) - len(olid_train_clean)

print(f"  Removed {removed} texts < {min_length} chars")
print(f"  OLID Train: {len(olid_train)} -> {len(olid_train_clean)}")

# ===========================================
# 4. SAVE V5 DATASET
# ===========================================
print("\n[4/4] Saving V5 dataset...")

goemo_balanced.to_csv('data/kaggle_upload_v5/goemotions_balanced.csv', index=False)
olid_train_clean.to_csv('data/kaggle_upload_v5/olid_train.csv', index=False)
olid_val.to_csv('data/kaggle_upload_v5/olid_validation.csv', index=False)

print(f"  Saved: goemotions_balanced.csv ({len(goemo_balanced)} samples)")
print(f"  Saved: olid_train.csv ({len(olid_train_clean)} samples)")
print(f"  Saved: olid_validation.csv ({len(olid_val)} samples)")

# ===========================================
# FINAL SUMMARY
# ===========================================
print("\n" + "="*60)
print("V5 DATA READY")
print("="*60)
print(f"""
Files created in: data/kaggle_upload_v5/

Changes from V4:
  - GoEmotions: Balanced emotions (Disgust 3x, Fear 3x, Joy/Neutral reduced)
  - OLID: Removed {removed} short/noisy texts

Expected improvement:
  - Better Disgust/Fear learning -> Higher OFF Recall
  - Less noise -> Lower Gap

Next: Upload data/kaggle_upload_v5/ to Kaggle as 'aura-data-v5'
""")
