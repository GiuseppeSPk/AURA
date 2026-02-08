"""
AURA Data Preprocessing Script V2
Uses Hugging Face datasets library for GoEmotions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("🧹 AURA DATA PREPROCESSING V2")
print("="*60)

# Install datasets if needed
try:
    from datasets import load_dataset
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "datasets", "-q"])
    from datasets import load_dataset

# Paths
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. GOEMOTIONS - Load from Hugging Face
# ============================================================
print("\n📥 [1/3] Loading GoEmotions from Hugging Face...")

dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
print(f"   ✅ Loaded! Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

# Combine all splits
all_data = []
for split in ['train', 'validation', 'test']:
    df = dataset[split].to_pandas()
    all_data.append(df)

goemo_raw = pd.concat(all_data, ignore_index=True)
print(f"   📊 Total samples: {len(goemo_raw)}")

# ============================================================
# 2. Map to 7 Core Emotions (Ekman + Neutral)
# ============================================================
print("\n🔄 [2/3] Mapping to 7 core emotions...")

# GoEmotions simplified has 'labels' column with list of label indices
# Label mapping from the dataset
LABEL_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

# Ekman mapping
EKMAN_MAPPING = {
    # Anger cluster
    'anger': 'anger',
    'annoyance': 'anger',
    'disapproval': 'anger',
    
    # Disgust
    'disgust': 'disgust',
    
    # Fear cluster
    'fear': 'fear',
    'nervousness': 'fear',
    
    # Joy cluster (positive emotions)
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

# Process
goemo_clean = pd.DataFrame()
goemo_clean['text'] = goemo_raw['text']

# Initialize emotion columns
for emo in CORE_EMOTIONS:
    goemo_clean[emo] = 0

# Map labels
for idx, row in goemo_raw.iterrows():
    labels = row['labels']
    if isinstance(labels, (list, np.ndarray)):
        for label_idx in labels:
            if label_idx < len(LABEL_NAMES):
                original_emotion = LABEL_NAMES[label_idx]
                if original_emotion in EKMAN_MAPPING:
                    core_emotion = EKMAN_MAPPING[original_emotion]
                    goemo_clean.at[idx, core_emotion] = 1

# Filter: Keep only rows where at least one emotion is present
has_emotion = goemo_clean[CORE_EMOTIONS].sum(axis=1) > 0
goemo_filtered = goemo_clean[has_emotion].copy()

print(f"\n   📊 After filtering:")
print(f"   Before: {len(goemo_clean):,} → After: {len(goemo_filtered):,}")
print(f"\n   Emotion distribution:")
for emo in CORE_EMOTIONS:
    count = int(goemo_filtered[emo].sum())
    pct = count / len(goemo_filtered) * 100
    print(f"   {emo:12}: {count:6,} ({pct:.1f}%)")

# Save
output_path = PROCESSED_DIR / "goemotions_clean.csv"
goemo_filtered.to_csv(output_path, index=False)
print(f"\n   ✅ Saved: {output_path}")

# ============================================================
# 3. Verify OLID
# ============================================================
print("\n🔍 [3/3] Verifying OLID dataset...")

olid_train = pd.read_csv(PROCESSED_DIR / "olid_train.csv")
olid_val = pd.read_csv(PROCESSED_DIR / "olid_validation.csv")

print(f"   OLID Train: {len(olid_train):,} rows")
print(f"   OLID Val:   {len(olid_val):,} rows")

# Label distribution
print(f"\n   Label distribution (Train):")
print(f"   NOT (0): {(olid_train['label'] == 0).sum():,} ({(olid_train['label'] == 0).mean()*100:.1f}%)")
print(f"   OFF (1): {(olid_train['label'] == 1).sum():,} ({(olid_train['label'] == 1).mean()*100:.1f}%)")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE")
print("="*60)

print(f"\n📦 Clean files ready:")
print(f"   - goemotions_clean.csv: {len(goemo_filtered):,} rows")
print(f"   - olid_train.csv: {len(olid_train):,} rows")  
print(f"   - olid_validation.csv: {len(olid_val):,} rows")

print("\n🎯 Next: Upload goemotions_clean.csv to Kaggle and retrain!")
