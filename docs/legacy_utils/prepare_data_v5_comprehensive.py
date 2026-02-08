"""
================================================================================
AURA V5 COMPREHENSIVE DATA PIPELINE
================================================================================
This script prepares production-quality training data by applying rigorous
filtering and balancing to all datasets.

Datasets:
- OLID (Toxicity): Train + Validation
- GoEmotions (Emotions): Filtered + Balanced

Filters Applied:
1. OLID: Remove short texts (<25 chars)
2. GoEmotions: Keep only single-label samples
3. GoEmotions: Keep only toxicity-relevant emotions (anger, disgust, fear, joy, neutral)
4. GoEmotions: Balance classes via under/oversampling
5. Validation: Remove any train/val overlap

Author: AURA Team
Date: 2026-01-18
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
from collections import Counter

# Configuration
CONFIG = {
    'min_text_length': 25,  # Minimum characters for OLID
    'min_goemo_length': 20,  # Minimum characters for GoEmotions
    'useful_emotions': ['anger', 'disgust', 'fear', 'joy', 'neutral'],  # Toxicity-relevant
    'balance_targets': {
        'anger': 5000,    # Strong toxicity signal
        'disgust': 3000,  # Strong toxicity signal (oversample from 1013)
        'fear': 2500,     # Moderate signal (oversample from 929)
        'joy': 6000,      # Anti-toxicity signal (undersample from 21k)
        'neutral': 6000   # Baseline (undersample from 17k)
    },
    'output_dir': 'data/kaggle_upload_v5'
}

print("="*70)
print("AURA V5 COMPREHENSIVE DATA PIPELINE")
print("="*70)

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# STEP 1: LOAD RAW DATA
# ============================================================================
print("\n" + "-"*70)
print("STEP 1: LOADING RAW DATA")
print("-"*70)

# Load all datasets
goemo_raw = pd.read_csv('data/kaggle_upload_v2/goemotions_clean.csv')
olid_train_raw = pd.read_csv('data/kaggle_upload_v2/olid_train.csv')
olid_val_raw = pd.read_csv('data/kaggle_upload_v2/olid_validation.csv')

print(f"GoEmotions Raw: {len(goemo_raw):,} samples")
print(f"OLID Train Raw: {len(olid_train_raw):,} samples")
print(f"OLID Val Raw:   {len(olid_val_raw):,} samples")

# ============================================================================
# STEP 2: CLEAN OLID
# ============================================================================
print("\n" + "-"*70)
print("STEP 2: CLEANING OLID")
print("-"*70)

# 2a. Remove short texts
olid_train = olid_train_raw[olid_train_raw['text'].str.len() >= CONFIG['min_text_length']].copy()
removed_short = len(olid_train_raw) - len(olid_train)
print(f"Removed {removed_short} short texts (<{CONFIG['min_text_length']} chars)")

# 2b. Remove exact duplicates
before_dedup = len(olid_train)
olid_train = olid_train.drop_duplicates(subset=['text'])
removed_dup = before_dedup - len(olid_train)
print(f"Removed {removed_dup} duplicate texts")

# 2c. Check train/val overlap
train_texts = set(olid_train['text'].str.lower().str.strip())
val_texts = set(olid_val_raw['text'].str.lower().str.strip())
overlap = train_texts.intersection(val_texts)
print(f"Train/Val overlap: {len(overlap)} texts")

# Remove overlapping texts from validation
olid_val = olid_val_raw[~olid_val_raw['text'].str.lower().str.strip().isin(overlap)].copy()
print(f"Removed {len(overlap)} overlapping texts from validation")

# Final OLID stats
print(f"\nOLID Train Final: {len(olid_train):,} samples")
print(f"OLID Val Final:   {len(olid_val):,} samples")

# Label distribution
print("\nLabel Distribution:")
print("  Train:", dict(olid_train['label'].value_counts()))
print("  Val:  ", dict(olid_val['label'].value_counts()))

# ============================================================================
# STEP 3: FILTER GOEMOTIONS
# ============================================================================
print("\n" + "-"*70)
print("STEP 3: FILTERING GOEMOTIONS")
print("-"*70)

all_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
useful = CONFIG['useful_emotions']

# 3a. Filter to minimum text length
goemo = goemo_raw[goemo_raw['text'].str.len() >= CONFIG['min_goemo_length']].copy()
print(f"After length filter: {len(goemo):,} samples")

# 3b. Keep only single-label samples (reduce noise from multi-label ambiguity)
goemo['num_labels'] = goemo[all_emotions].sum(axis=1)
goemo_single = goemo[goemo['num_labels'] == 1].copy()
print(f"After single-label filter: {len(goemo_single):,} samples ({len(goemo) - len(goemo_single)} multi-label removed)")

# 3c. Keep only useful emotions
goemo_useful = goemo_single[goemo_single[useful].sum(axis=1) == 1].copy()
removed_emotions = len(goemo_single) - len(goemo_useful)
print(f"After emotion filter: {len(goemo_useful):,} samples ({removed_emotions} non-useful emotions removed)")

# Show current distribution
print("\nCurrent Distribution (before balancing):")
for emo in useful:
    count = goemo_useful[emo].sum()
    pct = count / len(goemo_useful) * 100
    bar = '#' * int(pct / 3)
    print(f"  {emo:10}: {int(count):5} ({pct:5.1f}%) {bar}")

# ============================================================================
# STEP 4: BALANCE GOEMOTIONS
# ============================================================================
print("\n" + "-"*70)
print("STEP 4: BALANCING GOEMOTIONS")
print("-"*70)

balanced_dfs = []
for emo in useful:
    target = CONFIG['balance_targets'][emo]
    emo_data = goemo_useful[goemo_useful[emo] == 1].copy()
    current = len(emo_data)
    
    if current == 0:
        print(f"  {emo:10}: SKIPPED (no samples)")
        continue
    
    if target <= current:
        # Undersample
        sampled = emo_data.sample(target, random_state=42)
        action = "undersampled"
    else:
        # Oversample with replacement
        sampled = resample(emo_data, n_samples=target, replace=True, random_state=42)
        action = "oversampled"
    
    balanced_dfs.append(sampled)
    print(f"  {emo:10}: {current:5} -> {target:5} ({action})")

# Combine balanced samples
goemo_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Remove exact duplicates (from oversampling, keep one copy)
goemo_balanced = goemo_balanced.drop_duplicates(subset=['text'])

print(f"\nGoEmotions Balanced Final: {len(goemo_balanced):,} samples")

# Final distribution
print("\nFinal Distribution:")
total = len(goemo_balanced)
for emo in useful:
    count = goemo_balanced[emo].sum()
    pct = count / total * 100
    bar = '#' * int(pct / 3)
    print(f"  {emo:10}: {int(count):5} ({pct:5.1f}%) {bar}")

# ============================================================================
# STEP 5: VALIDATE DATA QUALITY
# ============================================================================
print("\n" + "-"*70)
print("STEP 5: DATA QUALITY VALIDATION")
print("-"*70)

checks_passed = 0
checks_total = 6

# Check 1: No empty texts
empty_olid = olid_train['text'].isna().sum() + olid_train['text'].str.strip().eq('').sum()
empty_goemo = goemo_balanced['text'].isna().sum()
if empty_olid == 0 and empty_goemo == 0:
    print("[PASS] No empty texts")
    checks_passed += 1
else:
    print(f"[FAIL] Empty texts: OLID={empty_olid}, GoEmo={empty_goemo}")

# Check 2: All labels valid
olid_labels = set(olid_train['label'].unique())
if olid_labels == {0, 1}:
    print("[PASS] OLID labels valid (0, 1)")
    checks_passed += 1
else:
    print(f"[FAIL] OLID labels invalid: {olid_labels}")

# Check 3: No train/val overlap
overlap_check = len(set(olid_train['text']).intersection(set(olid_val['text'])))
if overlap_check == 0:
    print("[PASS] No train/val overlap")
    checks_passed += 1
else:
    print(f"[FAIL] Train/val overlap: {overlap_check}")

# Check 4: Emotions are binary
emo_values = goemo_balanced[useful].values
if set(np.unique(emo_values)) == {0, 1} or set(np.unique(emo_values)) == {0.0, 1.0}:
    print("[PASS] Emotion labels are binary")
    checks_passed += 1
else:
    print(f"[FAIL] Non-binary emotion values found")

# Check 5: No all-zero emotion rows
zero_rows = (goemo_balanced[useful].sum(axis=1) == 0).sum()
if zero_rows == 0:
    print("[PASS] No all-zero emotion rows")
    checks_passed += 1
else:
    print(f"[FAIL] All-zero rows: {zero_rows}")

# Check 6: Reasonable class balance
label_counts = olid_train['label'].value_counts()
ratio = label_counts[0] / label_counts[1]
if 1.5 <= ratio <= 3.0:
    print(f"[PASS] OLID class ratio: {ratio:.2f}:1 (acceptable)")
    checks_passed += 1
else:
    print(f"[WARN] OLID class ratio: {ratio:.2f}:1 (may need weighting)")
    checks_passed += 0.5

print(f"\nValidation: {checks_passed}/{checks_total} checks passed")

# ============================================================================
# STEP 6: SAVE FINAL DATASETS
# ============================================================================
print("\n" + "-"*70)
print("STEP 6: SAVING FINAL DATASETS")
print("-"*70)

# Keep only necessary columns
goemo_final = goemo_balanced[['text'] + useful].copy()
olid_train_final = olid_train[['text', 'label']].copy()
olid_val_final = olid_val[['text', 'label']].copy()

# Save
goemo_final.to_csv(f"{CONFIG['output_dir']}/goemotions_v5.csv", index=False)
olid_train_final.to_csv(f"{CONFIG['output_dir']}/olid_train.csv", index=False)
olid_val_final.to_csv(f"{CONFIG['output_dir']}/olid_validation.csv", index=False)

print(f"Saved: goemotions_v5.csv     ({len(goemo_final):,} samples)")
print(f"Saved: olid_train.csv        ({len(olid_train_final):,} samples)")
print(f"Saved: olid_validation.csv   ({len(olid_val_final):,} samples)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("V5 DATA PIPELINE COMPLETE")
print("="*70)
print(f"""
OUTPUT: {CONFIG['output_dir']}/

DATASETS:
  GoEmotions V5:     {len(goemo_final):,} samples (5 emotions, balanced, single-label)
  OLID Train:        {len(olid_train_final):,} samples (noise filtered)
  OLID Validation:   {len(olid_val_final):,} samples (overlap removed)

IMPROVEMENTS OVER V4:
  - Removed multi-label ambiguity (single-label only)
  - Kept only toxicity-relevant emotions
  - Balanced emotion distribution
  - Removed short/noisy texts
  - Verified train/val disjoint

NEXT STEPS:
  1. Upload {CONFIG['output_dir']}/ to Kaggle as 'aura-data-v5'
  2. Update notebook to use 'goemotions_v5.csv'
  3. Update notebook CONFIG: 'num_emotion_classes': 5
  4. Run training
""")
