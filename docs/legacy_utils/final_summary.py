# Final summary
import pandas as pd
from pathlib import Path

DST = Path('data/kaggle_upload_v2')

print('=== FINAL V2 DATASET SUMMARY ===')
print()

# OLID Train
olid_train = pd.read_csv(DST / 'olid_train.csv')
not_count = (olid_train['label']==0).sum()
off_count = (olid_train['label']==1).sum()
print(f'olid_train.csv: {len(olid_train)} rows')
print(f'  NOT: {not_count}, OFF: {off_count}')

# OLID Val
olid_val = pd.read_csv(DST / 'olid_validation.csv')
print(f'olid_validation.csv: {len(olid_val)} rows')

# Overlap check
train_set = set(olid_train['text'].str.lower().str.strip())
val_set = set(olid_val['text'].str.lower().str.strip())
overlap = len(train_set & val_set)
print(f'  Overlap with train: {overlap}')

# GoEmotions
goemo = pd.read_csv(DST / 'goemotions_clean.csv')
print(f'goemotions_clean.csv: {len(goemo)} rows')
emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
for col in emo_cols:
    print(f'  {col}: {goemo[col].sum()}')

print()
if overlap == 0:
    print('=== ALL CHECKS PASSED - READY TO UPLOAD ===')
else:
    print('=== ISSUES FOUND ===')
