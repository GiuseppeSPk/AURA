"""
AURA Pre-Training Validation Suite
Comprehensive tests before Kaggle training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from collections import Counter

print("="*70)
print("🔬 AURA PRE-TRAINING VALIDATION SUITE")
print("="*70)

PROCESSED_DIR = Path("data/processed")
CORE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']

tests_passed = 0
tests_failed = 0

def test_result(name, passed, details=""):
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        print(f"   ✅ {name}")
    else:
        tests_failed += 1
        print(f"   ❌ {name}: {details}")
    return passed

# ============================================================
# TEST 1: File Existence
# ============================================================
print("\n📁 [TEST 1] File Existence")
print("-" * 50)

goemo_path = PROCESSED_DIR / "goemotions_clean.csv"
olid_train_path = PROCESSED_DIR / "olid_train.csv"
olid_val_path = PROCESSED_DIR / "olid_validation.csv"

test_result("goemotions_clean.csv exists", goemo_path.exists())
test_result("olid_train.csv exists", olid_train_path.exists())
test_result("olid_validation.csv exists", olid_val_path.exists())

# Load data
goemo = pd.read_csv(goemo_path)
olid_train = pd.read_csv(olid_train_path)
olid_val = pd.read_csv(olid_val_path)

# ============================================================
# TEST 2: Column Structure
# ============================================================
print("\n📋 [TEST 2] Column Structure")
print("-" * 50)

expected_goemo_cols = ['text'] + CORE_EMOTIONS
expected_olid_cols = ['text', 'label']

test_result("GoEmotions has 'text' column", 'text' in goemo.columns)
test_result("GoEmotions has all 7 emotion columns", all(c in goemo.columns for c in CORE_EMOTIONS))
test_result("OLID Train has 'text' column", 'text' in olid_train.columns)
test_result("OLID Train has 'label' column", 'label' in olid_train.columns)

# ============================================================
# TEST 3: No Missing Values
# ============================================================
print("\n🔍 [TEST 3] Missing Values")
print("-" * 50)

test_result("GoEmotions: no missing text", goemo['text'].isna().sum() == 0, 
            f"{goemo['text'].isna().sum()} missing")
test_result("OLID Train: no missing text", olid_train['text'].isna().sum() == 0,
            f"{olid_train['text'].isna().sum()} missing")
test_result("OLID Train: no missing labels", olid_train['label'].isna().sum() == 0,
            f"{olid_train['label'].isna().sum()} missing")

# ============================================================
# TEST 4: No Empty Strings
# ============================================================
print("\n📝 [TEST 4] Empty Strings")
print("-" * 50)

goemo_empty = (goemo['text'].str.strip() == '').sum()
olid_train_empty = (olid_train['text'].str.strip() == '').sum()

test_result("GoEmotions: no empty strings", goemo_empty == 0, f"{goemo_empty} empty")
test_result("OLID Train: no empty strings", olid_train_empty == 0, f"{olid_train_empty} empty")

# ============================================================
# TEST 5: Label Validity
# ============================================================
print("\n🏷️ [TEST 5] Label Validity")
print("-" * 50)

# GoEmotions: all emotion columns should be 0 or 1
for col in CORE_EMOTIONS:
    valid = goemo[col].isin([0, 1]).all()
    test_result(f"GoEmotions '{col}' is binary (0/1)", valid)

# OLID: labels should be 0 or 1
test_result("OLID labels are binary (0/1)", olid_train['label'].isin([0, 1]).all())

# ============================================================
# TEST 6: No All-Zero Emotion Rows
# ============================================================
print("\n⚠️ [TEST 6] All-Zero Emotion Rows")
print("-" * 50)

all_zero_rows = (goemo[CORE_EMOTIONS].sum(axis=1) == 0).sum()
test_result("GoEmotions: no all-zero emotion rows", all_zero_rows == 0, 
            f"{all_zero_rows} rows with all zeros")

# ============================================================
# TEST 7: All Emotion Classes Have Samples
# ============================================================
print("\n📊 [TEST 7] Emotion Class Distribution")
print("-" * 50)

for col in CORE_EMOTIONS:
    count = goemo[col].sum()
    test_result(f"'{col}' has samples", count > 0, f"count = {count}")
    print(f"      → {col}: {count:,} samples ({count/len(goemo)*100:.1f}%)")

# ============================================================
# TEST 8: OLID Class Balance
# ============================================================
print("\n⚖️ [TEST 8] OLID Class Balance")
print("-" * 50)

not_count = (olid_train['label'] == 0).sum()
off_count = (olid_train['label'] == 1).sum()
ratio = off_count / not_count

print(f"   NOT (non-toxic): {not_count:,} ({not_count/len(olid_train)*100:.1f}%)")
print(f"   OFF (toxic):     {off_count:,} ({off_count/len(olid_train)*100:.1f}%)")
print(f"   Ratio OFF/NOT:   {ratio:.2f}")

test_result("Class imbalance is manageable (<3:1)", ratio > 0.33, f"ratio = {ratio:.2f}")

# ============================================================
# TEST 9: Duplicate Detection
# ============================================================
print("\n🔄 [TEST 9] Duplicate Detection")
print("-" * 50)

goemo_dupes = goemo['text'].duplicated().sum()
olid_train_dupes = olid_train['text'].duplicated().sum()
olid_val_dupes = olid_val['text'].duplicated().sum()

# Allow some duplicates (different emotions for same text is valid)
test_result(f"GoEmotions duplicates < 10%", goemo_dupes < len(goemo) * 0.1, 
            f"{goemo_dupes} duplicates ({goemo_dupes/len(goemo)*100:.1f}%)")
test_result(f"OLID Train duplicates < 5%", olid_train_dupes < len(olid_train) * 0.05,
            f"{olid_train_dupes} duplicates")

# ============================================================
# TEST 10: Train/Val Overlap (Data Leakage)
# ============================================================
print("\n🚨 [TEST 10] Train/Val Data Leakage")
print("-" * 50)

train_texts = set(olid_train['text'].str.lower().str.strip())
val_texts = set(olid_val['text'].str.lower().str.strip())
overlap = train_texts.intersection(val_texts)

test_result("No train/val overlap", len(overlap) == 0, f"{len(overlap)} overlapping texts!")

# ============================================================
# TEST 11: Text Length Statistics
# ============================================================
print("\n📏 [TEST 11] Text Length Statistics")
print("-" * 50)

goemo_lengths = goemo['text'].str.len()
olid_lengths = olid_train['text'].str.len()

print(f"   GoEmotions - Mean: {goemo_lengths.mean():.0f}, Max: {goemo_lengths.max()}, Min: {goemo_lengths.min()}")
print(f"   OLID Train - Mean: {olid_lengths.mean():.0f}, Max: {olid_lengths.max()}, Min: {olid_lengths.min()}")

# Check for very short texts that might be problematic
very_short_goemo = (goemo_lengths < 10).sum()
very_short_olid = (olid_lengths < 10).sum()

test_result("GoEmotions: <5% very short texts (<10 chars)", very_short_goemo < len(goemo) * 0.05,
            f"{very_short_goemo} texts < 10 chars")
test_result("OLID: <5% very short texts (<10 chars)", very_short_olid < len(olid_train) * 0.05,
            f"{very_short_olid} texts < 10 chars")

# ============================================================
# TEST 12: Tokenization Test
# ============================================================
print("\n🔤 [TEST 12] Tokenization Test")
print("-" * 50)

try:
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Test on sample texts
    sample_texts = [
        goemo['text'].iloc[0],
        goemo['text'].iloc[100],
        olid_train['text'].iloc[0],
        olid_train['text'].iloc[100]
    ]
    
    for i, text in enumerate(sample_texts):
        tokens = tokenizer.encode(text, max_length=128, truncation=True)
        test_result(f"Sample {i+1} tokenizes correctly", len(tokens) > 2, f"only {len(tokens)} tokens")
    
except Exception as e:
    test_result("Tokenizer loads and works", False, str(e))

# ============================================================
# TEST 13: Model Forward Pass
# ============================================================
print("\n🧠 [TEST 13] Model Forward Pass Test")
print("-" * 50)

try:
    from transformers import DistilBertModel
    import torch.nn as nn
    
    class QuickModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
            self.tox_head = nn.Linear(768, 2)
            self.emo_head = nn.Linear(768, 7)
        
        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0]
            return self.tox_head(cls), self.emo_head(cls)
    
    model = QuickModel()
    
    # Forward pass with real data
    enc = tokenizer(
        [goemo['text'].iloc[0], olid_train['text'].iloc[0]],
        padding=True, truncation=True, max_length=128, return_tensors='pt'
    )
    
    with torch.no_grad():
        tox_out, emo_out = model(enc['input_ids'], enc['attention_mask'])
    
    test_result("Model forward pass works", tox_out.shape == (2, 2) and emo_out.shape == (2, 7))
    print(f"      → Tox output shape: {tox_out.shape}")
    print(f"      → Emo output shape: {emo_out.shape}")
    
except Exception as e:
    test_result("Model forward pass works", False, str(e))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("📊 FINAL VALIDATION SUMMARY")
print("="*70)

total_tests = tests_passed + tests_failed
print(f"\n   ✅ Passed: {tests_passed}/{total_tests}")
print(f"   ❌ Failed: {tests_failed}/{total_tests}")

if tests_failed == 0:
    print("\n   🎉 ALL TESTS PASSED! Data is ready for training.")
else:
    print(f"\n   ⚠️ {tests_failed} test(s) failed. Review issues before training.")

print("\n" + "="*70)
