"""Dataset Analysis - Safe UTF-8 version"""
import pandas as pd
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = Path(r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data')

results = {}

# 1. Toxicity Train
print("Loading toxicity_train.csv...")
tox_train = pd.read_csv(DATA_DIR / 'toxicity_train.csv', encoding='utf-8')
results['tox_train'] = {
    'shape': tox_train.shape,
    'columns': list(tox_train.columns),
    'label_dist': dict(tox_train['label'].value_counts()),
    'missing': int(tox_train.isnull().sum().sum()),
    'duplicates': int(tox_train.duplicated().sum())
}

# 2. Toxicity Val
print("Loading toxicity_val.csv...")
tox_val = pd.read_csv(DATA_DIR / 'toxicity_val.csv', encoding='utf-8')
results['tox_val'] = {
    'shape': tox_val.shape,
    'label_dist': dict(tox_val['label'].value_counts())
}

# 3. Emotions
print("Loading emotions_train.csv...")
emo_train = pd.read_csv(DATA_DIR / 'emotions_train.csv', encoding='utf-8')
emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
present_emo = [c for c in emo_cols if c in emo_train.columns]
results['emo_train'] = {
    'shape': emo_train.shape,
    'columns': list(emo_train.columns),
    'emo_cols_present': present_emo,
    'label_counts': {col: int(emo_train[col].sum()) for col in present_emo} if present_emo else {},
    'has_label_sum': 'label_sum' in emo_train.columns,
    'missing': int(emo_train.isnull().sum().sum())
}

if 'label_sum' in emo_train.columns:
    results['emo_train']['no_labels'] = int((emo_train['label_sum'] == 0).sum())
    results['emo_train']['multi_labels'] = int((emo_train['label_sum'] > 1).sum())

# 4. Sentiment
print("Loading sentiment_train.csv...")
sent_train = pd.read_csv(DATA_DIR / 'sentiment_train.csv', encoding='utf-8')
results['sent_train'] = {
    'shape': sent_train.shape,
    'columns': list(sent_train.columns),
    'label_dist': dict(sent_train['label'].value_counts()),
    'missing': int(sent_train.isnull().sum().sum())
}

# 5. Reporting
print("Loading reporting_examples.csv...")
rep_train = pd.read_csv(DATA_DIR / 'reporting_examples.csv', encoding='utf-8')
results['rep_train'] = {
    'shape': rep_train.shape,
    'columns': list(rep_train.columns),
    'all_rows': rep_train.to_dict('records')
}

if 'is_reporting' in rep_train.columns:
    results['rep_train']['reporting_dist'] = dict(rep_train['is_reporting'].value_counts())

# Print results
import json
print("\n" + "="*80)
print(json.dumps(results, indent=2, ensure_ascii=False))
print("="*80)
