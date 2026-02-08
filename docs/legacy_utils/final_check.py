# Final data check
import pandas as pd
from pathlib import Path

print('='*60)
print('FINAL DATA CHECK - kaggle_upload_v2')
print('='*60)

DST = Path('data/kaggle_upload_v2')

# 1. OLID TRAIN
print('\n[1] olid_train.csv')
df = pd.read_csv(DST / 'olid_train.csv')
print(f'    Rows: {len(df)}')
not_count = (df['label']==0).sum()
off_count = (df['label']==1).sum()
print(f'    NOT (0): {not_count}')
print(f'    OFF (1): {off_count}')
print(f'    Missing text: {df["text"].isna().sum()}')

# 2. OLID VALIDATION  
print('\n[2] olid_validation.csv')
df_val = pd.read_csv(DST / 'olid_validation.csv')
print(f'    Rows: {len(df_val)}')

# Overlap check
train_set = set(df['text'].str.lower().str.strip())
val_set = set(df_val['text'].str.lower().str.strip())
overlap = len(train_set & val_set)
print(f'    Overlap with train: {overlap}')

# 3. GOEMOTIONS
print('\n[3] goemotions_clean.csv')
goemo = pd.read_csv(DST / 'goemotions_clean.csv')
print(f'    Rows: {len(goemo)}')
emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
print('    Emotion counts:')
for col in emo_cols:
    count = int(goemo[col].sum())
    status = 'OK' if count > 0 else 'BROKEN!'
    print(f'      {col}: {count} [{status}]')

# All-zero check
all_zero = (goemo[emo_cols].sum(axis=1) == 0).sum()
print(f'    All-zero rows: {all_zero}')

# Final verdict
print('\n' + '='*60)
issues = []
if overlap > 0: issues.append('Train/Val overlap')
if goemo['disgust'].sum() == 0: issues.append('Disgust empty')
if goemo['neutral'].sum() == 0: issues.append('Neutral empty')
if all_zero > 0: issues.append('GoEmo has all-zero rows')

if not issues:
    print('ALL CHECKS PASSED - READY FOR KAGGLE')
else:
    print(f'ISSUES: {issues}')
print('='*60)
