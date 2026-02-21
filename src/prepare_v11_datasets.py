"""
AURA V11 — Data Preparation Script
===================================
Downloads official train/dev splits for GoEmotions and SST-2,
then saves them as CSVs matching the column format used in v10.2.

Requirements:
    pip install datasets pandas

Output files (saved to --output_dir):
    emotions_train.csv   - GoEmotions train split, Ekman-mapped (7 binary cols)
    emotions_val.csv     - GoEmotions dev split, Ekman-mapped
    sentiment_train.csv  - SST-2 train split (text, label)
    sentiment_val.csv    - SST-2 dev split (text, label)

Usage:
    python prepare_v11_datasets.py --output_dir ./data
"""

import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# GoEmotions Ekman mapping (official, from Google Research)
# Maps 28 fine-grained emotions -> 7 Ekman categories
EKMAN_MAPPING = {
    'anger':    ['anger', 'annoyance', 'disapproval'],
    'disgust':  ['disgust'],
    'fear':     ['fear', 'nervousness'],
    'joy':      ['joy', 'amusement', 'approval', 'excitement', 'gratitude',
                 'love', 'optimism', 'relief', 'pride', 'admiration', 'desire', 'caring'],
    'sadness':  ['sadness', 'disappointment', 'embarrassment', 'grief', 'remorse'],
    'surprise': ['surprise', 'realization', 'confusion', 'curiosity'],
    'neutral':  ['neutral'],
}

EKMAN_COLS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

# GoEmotions original 28 label names (index-ordered)
GO_EMOTIONS_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Build reverse mapping: fine_label_index -> ekman_col_index
FINE_TO_EKMAN = {}
for ekman_idx, ekman_name in enumerate(EKMAN_COLS):
    for fine_name in EKMAN_MAPPING[ekman_name]:
        fine_idx = GO_EMOTIONS_LABELS.index(fine_name)
        FINE_TO_EKMAN[fine_idx] = ekman_idx


def process_go_emotions(split_name: str) -> pd.DataFrame:
    """Download a GoEmotions split and map to Ekman binary columns.
    
    Uses the 'simplified' config which has train/validation/test splits.
    Each sample has a 'labels' field with indices into the 28-emotion list.
    We map these to the 7 Ekman categories using FINE_TO_EKMAN.
    """
    print(f"  Loading GoEmotions '{split_name}' split...")
    ds = load_dataset('go_emotions', 'simplified', split=split_name)
    
    rows = []
    for sample in ds:
        text = sample['text']
        ekman = [0] * 7
        for fine_idx in sample['labels']:
            if fine_idx in FINE_TO_EKMAN:
                ekman[FINE_TO_EKMAN[fine_idx]] = 1
        label_sum = sum(ekman)
        rows.append([text] + ekman + [label_sum])
    
    df = pd.DataFrame(rows, columns=['text'] + EKMAN_COLS + ['label_sum'])
    print(f"    -> {len(df)} samples, {(df['label_sum'] > 0).sum()} with >=1 Ekman label")
    return df


def process_sst2(split_name: str) -> pd.DataFrame:
    """Download an SST-2 split and format to (text, label)."""
    print(f"  Loading SST-2 '{split_name}' split...")
    ds = load_dataset('glue', 'sst2', split=split_name)
    
    df = pd.DataFrame({
        'text': ds['sentence'],
        'label': ds['label']
    })
    print(f"    -> {len(df)} samples, label distribution: {dict(df['label'].value_counts().sort_index())}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Prepare AURA V11 datasets')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save CSV files (default: current dir)')
    args = parser.parse_args()
    
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AURA V11 - Dataset Preparation")
    print("=" * 60)
    
    # --- GoEmotions ---
    print("\n[GoEmotions] (Ekman mapping):")
    emo_train = process_go_emotions('train')
    emo_val = process_go_emotions('validation')
    
    emo_train.to_csv(out / 'emotions_train.csv', index=False)
    emo_val.to_csv(out / 'emotions_val.csv', index=False)
    print(f"  [OK] Saved emotions_train.csv ({len(emo_train)} rows)")
    print(f"  [OK] Saved emotions_val.csv ({len(emo_val)} rows)")
    
    # --- SST-2 ---
    print("\n[SST-2]:")
    sent_train = process_sst2('train')
    sent_val = process_sst2('validation')
    
    sent_train.to_csv(out / 'sentiment_train.csv', index=False)
    sent_val.to_csv(out / 'sentiment_val.csv', index=False)
    print(f"  [OK] Saved sentiment_train.csv ({len(sent_train)} rows)")
    print(f"  [OK] Saved sentiment_val.csv ({len(sent_val)} rows)")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Emotions train: {len(emo_train):>6,} samples")
    print(f"  Emotions val:   {len(emo_val):>6,} samples")
    print(f"  Sentiment train:{len(sent_train):>6,} samples")
    print(f"  Sentiment val:  {len(sent_val):>6,} samples")
    print()
    print("Copy these 4 files + toxicity_train.csv, toxicity_val.csv,")
    print("reporting_examples_augmented.csv to your Kaggle dataset 'aura-v11-data'.")
    print("=" * 60)


if __name__ == '__main__':
    main()
