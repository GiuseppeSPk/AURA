"""
AURA V9 - Analisi Completa Dataset per 3 Task Heads
"""
import pandas as pd
import os

print("=" * 70)
print("ANALISI DATASET PER AURA V9 - 3 TASK HEADS")
print("=" * 70)

# ============= 1. GOEMOTIONS RAW =============
print("\n" + "=" * 70)
print("1. GOEMOTIONS RAW (Emotions)")
print("=" * 70)

ge = pd.read_csv(r'data/raw/goemotions_raw.csv')
print(f"Righe totali: {len(ge):,}")

# Label columns (escludendo metadata)
meta_cols = ['text', 'id', 'author', 'subreddit', 'link_id', 'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']
emotion_cols = [c for c in ge.columns if c not in meta_cols]
print(f"Emozioni disponibili ({len(emotion_cols)}): {emotion_cols[:10]}...")

# Distribuzione top emozioni
print("\nDistribuzione emozioni (top 10):")
emotion_counts = [(col, ge[col].sum()) for col in emotion_cols]
emotion_counts.sort(key=lambda x: -x[1])
for col, count in emotion_counts[:10]:
    pct = count / len(ge) * 100
    print(f"  {col:20}: {count:,} ({pct:.1f}%)")

# Righe con zero label
zero_labels = (ge[emotion_cols].sum(axis=1) == 0).sum()
print(f"\nRighe con ZERO label: {zero_labels:,} ({zero_labels/len(ge)*100:.1f}%)")
print(f"Righe con almeno 1 label: {len(ge) - zero_labels:,}")

# ============= 2. OLID (TOXICITY) =============
print("\n" + "=" * 70)
print("2. OLID (Toxicity)")
print("=" * 70)

olid_train = pd.read_csv(r'data/processed/olid_train.csv')
olid_val = pd.read_csv(r'data/processed/olid_validation.csv')

print(f"Train: {len(olid_train):,} righe")
print(f"Validation: {len(olid_val):,} righe")
print(f"Colonne: {list(olid_train.columns)}")

# Distribuzione
non_toxic = (olid_train['label'] == 0).sum()
toxic = (olid_train['label'] == 1).sum()
print(f"\nDistribuzione Train:")
print(f"  Non-toxic (0): {non_toxic:,} ({non_toxic/len(olid_train)*100:.1f}%)")
print(f"  Toxic (1): {toxic:,} ({toxic/len(olid_train)*100:.1f}%)")

# Lunghezza testi
olid_train['text_len'] = olid_train['text'].str.len()
print(f"\nLunghezza testi (train):")
print(f"  Media: {olid_train['text_len'].mean():.0f} caratteri")
print(f"  Min: {olid_train['text_len'].min()}, Max: {olid_train['text_len'].max()}")

# ============= 3. SENTIMENT =============
print("\n" + "=" * 70)
print("3. SENTIMENT")
print("=" * 70)

sent = pd.read_csv(r'data/kaggle_mega/sentiment_train.csv')
print(f"Righe totali: {len(sent):,}")
print(f"Colonne: {list(sent.columns)}")

# Distribuzione
neg = (sent['label'] == 0).sum()
pos = (sent['label'] == 1).sum()
print(f"\nDistribuzione:")
print(f"  Negative (0): {neg:,} ({neg/len(sent)*100:.1f}%)")
print(f"  Positive (1): {pos:,} ({pos/len(sent)*100:.1f}%)")

# Sample
print("\nEsempi (primi 3):")
for i, row in sent.head(3).iterrows():
    label = "NEG" if row['label'] == 0 else "POS"
    text = str(row['text'])[:70]
    print(f"  [{label}] {text}...")

# Lunghezza
sent['text_len'] = sent['text'].str.len()
print(f"\nLunghezza testi:")
print(f"  Media: {sent['text_len'].mean():.0f} caratteri")
print(f"  Min: {sent['text_len'].min()}, Max: {sent['text_len'].max()}")

# ============= RIEPILOGO =============
print("\n" + "=" * 70)
print("RIEPILOGO - IDONEITA' DATASET PER AURA V9")
print("=" * 70)

print("""
TASK HEAD      | DATASET       | RIGHE    | BILANCIAMENTO | QUALITA'
---------------|---------------|----------|---------------|----------
Toxicity       | OLID          | ~12k     | 67/33%        | ★★★★★
Emotions       | GoEmotions    | ~54k*    | Multi-label   | ★★★★★
Sentiment      | SST-style     | ~73k     | ~50/50%       | ★★★★☆

* GoEmotions dopo filtro righe con almeno 1 label

RACCOMANDAZIONI:
1. OLID: Dataset eccellente per toxicity, ben bilanciato
2. GoEmotions: Gold standard per emozioni, richiede mapping Ekman
3. Sentiment: Verificare fonte (SST-2 consigliato)
""")
