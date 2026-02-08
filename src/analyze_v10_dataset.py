"""
AURA V10 - Dataset Analysis Script
===================================
Analizza tutti i dataset per verificare qualità e conformità.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Path al dataset
DATA_DIR = Path(r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data')

print("="*80)
print("AURA V10 - DATASET ANALYSIS")
print("="*80)

# 1. Toxicity Train
print("\n" + "="*80)
print("1. TOXICITY TRAIN (OLID)")
print("="*80)
tox_train = pd.read_csv(DATA_DIR / 'toxicity_train.csv')
print(f"Shape: {tox_train.shape}")
print(f"Columns: {list(tox_train.columns)}")
print(f"\nFirst 3 rows:")
print(tox_train.head(3))
print(f"\nLabel distribution:")
print(tox_train['label'].value_counts().sort_index())
print(f"Class balance: {tox_train['label'].value_counts(normalize=True).sort_index()}")
print(f"Missing values: {tox_train.isnull().sum().sum()}")
print(f"Duplicates: {tox_train.duplicated().sum()}")
print(f"Text length stats:")
print(tox_train['text'].str.len().describe())

# 2. Toxicity Val
print("\n" + "="*80)
print("2. TOXICITY VALIDATION")
print("="*80)
tox_val = pd.read_csv(DATA_DIR / 'toxicity_val.csv')
print(f"Shape: {tox_val.shape}")
print(f"Label distribution:")
print(tox_val['label'].value_counts().sort_index())
print(f"Split ratio (train/val): {len(tox_train)}/{len(tox_val)} = {len(tox_train)/len(tox_val):.2f}:1")

# 3. Emotions
print("\n" + "="*80)
print("3. EMOTIONS (GoEmotions → Ekman)")
print("="*80)
emo_train = pd.read_csv(DATA_DIR / 'emotions_train.csv')
print(f"Shape: {emo_train.shape}")
print(f"Columns: {list(emo_train.columns)}")
print(f"\nFirst 3 rows:")
print(emo_train.head(3))

expected_emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
present_emo_cols = [c for c in expected_emo_cols if c in emo_train.columns]
print(f"\nEmotion columns present: {present_emo_cols}")

if present_emo_cols:
    print(f"\nLabel distribution (counts):")
    for col in present_emo_cols:
        print(f"  {col:10s}: {emo_train[col].sum()}")
    
    if 'label_sum' in emo_train.columns:
        print(f"\nMulti-label stats:")
        print(emo_train['label_sum'].value_counts().sort_index())
        print(f"Samples with no labels: {(emo_train['label_sum'] == 0).sum()}")
        print(f"Samples with multiple labels: {(emo_train['label_sum'] > 1).sum()}")
    
    print(f"\nMissing values: {emo_train.isnull().sum().sum()}")
    print(f"Duplicates: {emo_train.duplicated().sum()}")

# 4. Sentiment
print("\n" + "="*80)
print("4. SENTIMENT (SST-2)")
print("="*80)
sent_train = pd.read_csv(DATA_DIR / 'sentiment_train.csv')
print(f"Shape: {sent_train.shape}")
print(f"Columns: {list(sent_train.columns)}")
print(f"\nFirst 3 rows:")
print(sent_train.head(3))
print(f"\nLabel distribution:")
print(sent_train['label'].value_counts().sort_index())
print(f"Class balance: {sent_train['label'].value_counts(normalize=True).sort_index()}")
print(f"Missing values: {sent_train.isnull().sum().sum()}")
print(f"Duplicates: {sent_train.duplicated().sum()}")

# 5. Reporting
print("\n" + "="*80)
print("5. REPORTING (Sprugnoli - Event Detection)")
print("="*80)
rep_train = pd.read_csv(DATA_DIR / 'reporting_examples.csv')
print(f"Shape: {rep_train.shape}")
print(f"Columns: {list(rep_train.columns)}")
print(f"\nAll rows:")
print(rep_train)

if 'is_reporting' in rep_train.columns:
    print(f"\nReporting distribution:")
    print(rep_train['is_reporting'].value_counts().sort_index())
else:
    print("\n⚠️ WARNING: 'is_reporting' column not found!")

# Summary
print("\n" + "="*80)
print("DATASET SUMMARY")
print("="*80)
print(f"Toxicity Train:    {len(tox_train):6,} samples")
print(f"Toxicity Val:      {len(tox_val):6,} samples")
print(f"Emotions:          {len(emo_train):6,} samples")
print(f"Sentiment:         {len(sent_train):6,} samples")
print(f"Reporting:         {len(rep_train):6,} samples")
print(f"TOTAL TRAIN:       {len(tox_train) + len(emo_train) + len(sent_train) + len(rep_train):6,} samples")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
