"""
AURA V9 - VALIDAZIONE FINALE PRE-TRAINING
==========================================
Check completo di tutti i dataset prima del training.
Non dare nulla per scontato.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 70)
print("AURA V9 - VALIDAZIONE FINALE PRE-TRAINING")
print("=" * 70)

DATA_DIR = Path("data/aura_v9_clean")
ERRORS = []
WARNINGS = []

def check_passed(msg):
    print(f"  ✅ {msg}")

def check_failed(msg):
    print(f"  ❌ {msg}")
    ERRORS.append(msg)

def check_warning(msg):
    print(f"  ⚠️  {msg}")
    WARNINGS.append(msg)

# ============================================================
# 1. FILE EXISTENCE
# ============================================================
print("\n[1/8] VERIFICA ESISTENZA FILE")
print("-" * 40)

required_files = [
    "emotions_train.csv",
    "toxicity_train.csv", 
    "toxicity_val.csv",
    "sentiment_train.csv"
]

for f in required_files:
    path = DATA_DIR / f
    if path.exists():
        check_passed(f"{f} esiste ({path.stat().st_size / 1024:.0f} KB)")
    else:
        check_failed(f"{f} NON TROVATO!")

# ============================================================
# 2. CARICAMENTO E STRUTTURA COLONNE
# ============================================================
print("\n[2/8] VERIFICA STRUTTURA COLONNE")
print("-" * 40)

# Emotions
emo = pd.read_csv(DATA_DIR / "emotions_train.csv")
expected_emo_cols = ['text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
if list(emo.columns) == expected_emo_cols:
    check_passed(f"Emotions: colonne corrette {expected_emo_cols}")
else:
    check_failed(f"Emotions: colonne errate! Attese {expected_emo_cols}, trovate {list(emo.columns)}")

# Toxicity
tox = pd.read_csv(DATA_DIR / "toxicity_train.csv")
if list(tox.columns) == ['text', 'label']:
    check_passed("Toxicity train: colonne corrette ['text', 'label']")
else:
    check_failed(f"Toxicity train: colonne errate! {list(tox.columns)}")

tox_val = pd.read_csv(DATA_DIR / "toxicity_val.csv")
if list(tox_val.columns) == ['text', 'label']:
    check_passed("Toxicity val: colonne corrette ['text', 'label']")
else:
    check_failed(f"Toxicity val: colonne errate! {list(tox_val.columns)}")

# Sentiment
sent = pd.read_csv(DATA_DIR / "sentiment_train.csv")
if list(sent.columns) == ['text', 'label']:
    check_passed("Sentiment: colonne corrette ['text', 'label']")
else:
    check_failed(f"Sentiment: colonne errate! {list(sent.columns)}")

# ============================================================
# 3. VALORI MANCANTI
# ============================================================
print("\n[3/8] VERIFICA VALORI MANCANTI")
print("-" * 40)

for name, df in [("Emotions", emo), ("Toxicity", tox), ("Toxicity Val", tox_val), ("Sentiment", sent)]:
    missing = df.isnull().sum().sum()
    if missing == 0:
        check_passed(f"{name}: nessun valore mancante")
    else:
        check_failed(f"{name}: {missing} valori mancanti!")
        print(f"      {df.isnull().sum()}")

# ============================================================
# 4. TIPI DI DATI
# ============================================================
print("\n[4/8] VERIFICA TIPI DI DATI")
print("-" * 40)

# Text deve essere string
for name, df in [("Emotions", emo), ("Toxicity", tox), ("Sentiment", sent)]:
    if df['text'].dtype == 'object':
        check_passed(f"{name} text: tipo string (object)")
    else:
        check_failed(f"{name} text: tipo errato {df['text'].dtype}")

# Labels emotions devono essere 0 o 1
emo_label_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
for col in emo_label_cols:
    unique = set(emo[col].unique())
    if unique.issubset({0, 1}):
        check_passed(f"Emotions {col}: valori binari (0/1)")
    else:
        check_failed(f"Emotions {col}: valori non binari! {unique}")

# Labels toxicity/sentiment devono essere 0 o 1
for name, df in [("Toxicity", tox), ("Sentiment", sent)]:
    unique = set(df['label'].unique())
    if unique == {0, 1}:
        check_passed(f"{name} label: binari (0, 1)")
    else:
        check_failed(f"{name} label: valori errati! {unique}")

# ============================================================
# 5. RIGHE VUOTE O PROBLEMATICHE
# ============================================================
print("\n[5/8] VERIFICA RIGHE PROBLEMATICHE")
print("-" * 40)

# Testi vuoti o troppo corti
for name, df in [("Emotions", emo), ("Toxicity", tox), ("Sentiment", sent)]:
    empty_text = (df['text'].str.strip() == '').sum()
    if empty_text == 0:
        check_passed(f"{name}: nessun testo vuoto")
    else:
        check_failed(f"{name}: {empty_text} testi vuoti!")
    
    short_text = (df['text'].str.len() < 5).sum()
    if short_text == 0:
        check_passed(f"{name}: nessun testo troppo corto (<5 char)")
    else:
        check_warning(f"{name}: {short_text} testi molto corti (<5 char)")

# Emotions: righe con tutte le label a 0
zero_labels = (emo[emo_label_cols].sum(axis=1) == 0).sum()
if zero_labels == 0:
    check_passed(f"Emotions: nessuna riga con zero labels")
else:
    check_failed(f"Emotions: {zero_labels} righe con TUTTE le label = 0!")

# ============================================================
# 6. DISTRIBUZIONE CLASSI
# ============================================================
print("\n[6/8] VERIFICA DISTRIBUZIONE CLASSI")
print("-" * 40)

# Toxicity
tox_pos = (tox['label'] == 1).mean() * 100
tox_neg = (tox['label'] == 0).mean() * 100
print(f"  Toxicity: Toxic {tox_pos:.1f}% / Non-toxic {tox_neg:.1f}%")
if 20 < tox_pos < 50:
    check_passed("Toxicity: distribuzione ragionevole (20-50% toxic)")
else:
    check_warning(f"Toxicity: distribuzione sbilanciata!")

# Sentiment
sent_pos = (sent['label'] == 1).mean() * 100
sent_neg = (sent['label'] == 0).mean() * 100
print(f"  Sentiment: Positive {sent_pos:.1f}% / Negative {sent_neg:.1f}%")
if 40 < sent_pos < 60:
    check_passed("Sentiment: distribuzione bilanciata (40-60%)")
else:
    check_warning("Sentiment: distribuzione sbilanciata!")

# Emotions
print("  Emotions (distribuzione label):")
for col in emo_label_cols:
    pct = emo[col].mean() * 100
    print(f"    {col:12}: {pct:.1f}%")

# ============================================================
# 7. DUPLICATI
# ============================================================
print("\n[7/8] VERIFICA DUPLICATI")
print("-" * 40)

for name, df in [("Emotions", emo), ("Toxicity", tox), ("Sentiment", sent)]:
    dups = df['text'].duplicated().sum()
    dup_pct = dups / len(df) * 100
    if dup_pct < 1:
        check_passed(f"{name}: {dups} duplicati ({dup_pct:.2f}%)")
    elif dup_pct < 5:
        check_warning(f"{name}: {dups} duplicati ({dup_pct:.1f}%)")
    else:
        check_failed(f"{name}: TROPPI duplicati {dups} ({dup_pct:.1f}%)!")

# ============================================================
# 8. OVERLAP TRA DATASET
# ============================================================
print("\n[8/8] VERIFICA OVERLAP TRA DATASET")
print("-" * 40)

emo_texts = set(emo['text'].str.lower().str.strip())
tox_texts = set(tox['text'].str.lower().str.strip())
sent_texts = set(sent['text'].str.lower().str.strip())

overlap_emo_tox = len(emo_texts & tox_texts)
overlap_emo_sent = len(emo_texts & sent_texts)
overlap_tox_sent = len(tox_texts & sent_texts)

if overlap_emo_tox < 100:
    check_passed(f"Emotions ∩ Toxicity: {overlap_emo_tox} overlap")
else:
    check_warning(f"Emotions ∩ Toxicity: {overlap_emo_tox} overlap!")

if overlap_emo_sent < 100:
    check_passed(f"Emotions ∩ Sentiment: {overlap_emo_sent} overlap")
else:
    check_warning(f"Emotions ∩ Sentiment: {overlap_emo_sent} overlap!")

if overlap_tox_sent < 100:
    check_passed(f"Toxicity ∩ Sentiment: {overlap_tox_sent} overlap")
else:
    check_warning(f"Toxicity ∩ Sentiment: {overlap_tox_sent} overlap!")

# ============================================================
# RIEPILOGO FINALE
# ============================================================
print("\n" + "=" * 70)
print("RIEPILOGO VALIDAZIONE")
print("=" * 70)

print(f"\n📊 STATISTICHE DATASET:")
print(f"  Emotions:     {len(emo):>8,} righe")
print(f"  Toxicity:     {len(tox):>8,} righe")
print(f"  Toxicity Val: {len(tox_val):>8,} righe")
print(f"  Sentiment:    {len(sent):>8,} righe")
print(f"  TOTALE:       {len(emo)+len(tox)+len(tox_val)+len(sent):>8,} righe")

if ERRORS:
    print(f"\n❌ ERRORI CRITICI: {len(ERRORS)}")
    for e in ERRORS:
        print(f"   - {e}")
else:
    print("\n✅ NESSUN ERRORE CRITICO!")

if WARNINGS:
    print(f"\n⚠️  WARNINGS: {len(WARNINGS)}")
    for w in WARNINGS:
        print(f"   - {w}")
else:
    print("\n✅ NESSUN WARNING!")

if not ERRORS:
    print("\n" + "=" * 70)
    print("✅ VALIDAZIONE COMPLETATA - DATASET PRONTI PER IL TRAINING!")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("❌ VALIDAZIONE FALLITA - CORREGGERE ERRORI PRIMA DEL TRAINING!")
    print("=" * 70)
    sys.exit(1)
