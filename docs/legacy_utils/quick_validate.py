# Quick validation
import pandas as pd

goemo = pd.read_csv('data/processed/goemotions_clean.csv')
olid = pd.read_csv('data/processed/olid_train.csv')
olid_val = pd.read_csv('data/processed/olid_validation.csv')

print("=== VALIDATION RESULTS ===")
print(f"GoEmotions rows: {len(goemo)}")
print(f"OLID train rows: {len(olid)}")
print(f"OLID val rows: {len(olid_val)}")

emo_cols = ['anger','disgust','fear','joy','sadness','surprise','neutral']
all_have_emo = (goemo[emo_cols].sum(axis=1) > 0).all()
print(f"All GoEmotions have at least 1 emotion: {all_have_emo}")

for col in emo_cols:
    print(f"  {col}: {goemo[col].sum()}")

# Overlap check
train_set = set(olid['text'].str.lower().str.strip())
val_set = set(olid_val['text'].str.lower().str.strip())
overlap = len(train_set & val_set)
print(f"Train/Val text overlap: {overlap}")

if all_have_emo and goemo['disgust'].sum() > 0 and goemo['neutral'].sum() > 0:
    print("\n=== ALL CRITICAL CHECKS PASSED ===")
else:
    print("\n=== ISSUES FOUND ===")
