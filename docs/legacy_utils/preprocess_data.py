"""
AURA Data Preprocessing Script
Properly cleans and prepares GoEmotions and OLID datasets.

Author: Giuseppe
Date: 2026-01-18
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os

print("="*60)
print("🧹 AURA DATA PREPROCESSING")
print("="*60)

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. GOEMOTIONS - Download and Process Correctly
# ============================================================
print("\n📥 [1/4] Downloading GoEmotions from source...")

GOEMOTIONS_URL = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_1.csv"
GOEMOTIONS_URL_2 = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_2.csv"
GOEMOTIONS_URL_3 = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_3.csv"

# Download all parts
dfs = []
for i, url in enumerate([GOEMOTIONS_URL, GOEMOTIONS_URL_2, GOEMOTIONS_URL_3], 1):
    print(f"   Downloading part {i}/3...")
    try:
        df = pd.read_csv(url)
        dfs.append(df)
        print(f"   ✅ Part {i}: {len(df)} rows")
    except Exception as e:
        print(f"   ❌ Failed to download part {i}: {e}")

if dfs:
    goemo_raw = pd.concat(dfs, ignore_index=True)
    print(f"\n   📊 Total GoEmotions raw: {len(goemo_raw)} rows")
else:
    print("   ❌ Could not download GoEmotions. Using local file if available.")
    goemo_raw = None

# ============================================================
# 2. GOEMOTIONS - Map to 7 Core Emotions (Ekman + Neutral)
# ============================================================
print("\n🔄 [2/4] Mapping GoEmotions to 7 core classes...")

# Ekman's 6 basic emotions + neutral
# GoEmotions has 27 classes, we map them to 7
EMOTION_MAPPING = {
    # Anger cluster
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    
    # Disgust cluster
    'disgust': 'disgust',
    
    # Fear cluster
    'fear': 'fear',
    'nervousness': 'fear',
    
    # Joy cluster
    'joy': 'joy',
    'amusement': 'joy',
    'approval': 'joy',
    'excitement': 'joy',
    'gratitude': 'joy',
    'love': 'joy',
    'optimism': 'joy',
    'relief': 'joy',
    'pride': 'joy',
    'admiration': 'joy',
    'desire': 'joy',
    'caring': 'joy',
    
    # Sadness cluster
    'sadness': 'sadness',
    'disappointment': 'sadness',
    'embarrassment': 'sadness',
    'grief': 'sadness',
    'remorse': 'sadness',
    
    # Surprise cluster
    'surprise': 'surprise',
    'realization': 'surprise',
    'confusion': 'surprise',
    'curiosity': 'surprise',
    
    # Neutral
    'neutral': 'neutral'
}

CORE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

if goemo_raw is not None:
    # Get original emotion columns (they're the columns after text, id, author, etc.)
    # GoEmotions format: text, id, author, [27 emotion columns], example_very_unclear
    emotion_cols = [col for col in goemo_raw.columns if col in EMOTION_MAPPING.keys()]
    print(f"   Found emotion columns: {len(emotion_cols)}")
    
    # Create mapped dataset
    goemo_clean = pd.DataFrame()
    goemo_clean['text'] = goemo_raw['text']
    
    # Map each core emotion
    for core_emo in CORE_EMOTIONS:
        source_cols = [col for col, target in EMOTION_MAPPING.items() if target == core_emo and col in emotion_cols]
        if source_cols:
            # If ANY of the source emotions is 1, the core emotion is 1
            goemo_clean[core_emo] = goemo_raw[source_cols].max(axis=1)
        else:
            goemo_clean[core_emo] = 0
    
    # Filter: Keep only rows where at least one emotion is present
    has_emotion = goemo_clean[CORE_EMOTIONS].sum(axis=1) > 0
    goemo_filtered = goemo_clean[has_emotion].copy()
    
    print(f"\n   📊 After filtering (at least 1 emotion):")
    print(f"   Before: {len(goemo_clean)} → After: {len(goemo_filtered)}")
    print(f"\n   Emotion distribution:")
    for emo in CORE_EMOTIONS:
        count = goemo_filtered[emo].sum()
        pct = count / len(goemo_filtered) * 100
        print(f"   {emo:12}: {count:6} ({pct:.1f}%)")
    
    # Save
    goemo_filtered.to_csv(PROCESSED_DIR / "goemotions_clean.csv", index=False)
    print(f"\n   ✅ Saved: {PROCESSED_DIR / 'goemotions_clean.csv'}")
else:
    print("   ⚠️ Skipping GoEmotions processing (no data)")

# ============================================================
# 3. OLID - Verify and Clean
# ============================================================
print("\n🔍 [3/4] Verifying OLID dataset...")

olid_train_path = PROCESSED_DIR / "olid_train.csv"
olid_val_path = PROCESSED_DIR / "olid_validation.csv"

if olid_train_path.exists():
    olid_train = pd.read_csv(olid_train_path)
    olid_val = pd.read_csv(olid_val_path)
    
    print(f"   OLID Train: {len(olid_train)} rows")
    print(f"   OLID Val:   {len(olid_val)} rows")
    
    # Check label distribution
    print(f"\n   Train label distribution:")
    if 'label' in olid_train.columns:
        print(olid_train['label'].value_counts())
    
    # Check for missing values
    missing_text = olid_train['text'].isna().sum()
    print(f"\n   Missing text values: {missing_text}")
    
    # Check for empty strings
    empty_text = (olid_train['text'].str.strip() == '').sum()
    print(f"   Empty text strings: {empty_text}")
    
    if missing_text == 0 and empty_text == 0:
        print("   ✅ OLID data is clean!")
    else:
        print("   ⚠️ OLID has issues - cleaning...")
        olid_train = olid_train[olid_train['text'].str.strip() != '']
        olid_train = olid_train.dropna(subset=['text'])
        olid_train.to_csv(PROCESSED_DIR / "olid_train_clean.csv", index=False)
        
        olid_val = olid_val[olid_val['text'].str.strip() != '']
        olid_val = olid_val.dropna(subset=['text'])
        olid_val.to_csv(PROCESSED_DIR / "olid_validation_clean.csv", index=False)
        print("   ✅ Saved cleaned versions")
else:
    print("   ❌ OLID files not found!")

# ============================================================
# 4. Summary Report
# ============================================================
print("\n" + "="*60)
print("📊 FINAL SUMMARY")
print("="*60)

print("\n✅ Files created in data/processed/:")
for f in PROCESSED_DIR.glob("*.csv"):
    size = f.stat().st_size / 1024 / 1024
    df = pd.read_csv(f)
    print(f"   {f.name}: {len(df):,} rows ({size:.2f} MB)")

print("\n🎯 Next steps:")
print("   1. Upload 'goemotions_clean.csv' to Kaggle")
print("   2. Update notebook to use new file")
print("   3. Retrain V3")
