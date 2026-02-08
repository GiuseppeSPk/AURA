import pandas as pd
import numpy as np

print('='*60)
print('ANALISI APPROFONDITA DEI DATI')
print('='*60)

# OLID Train
olid_train = pd.read_csv('data/kaggle_upload_v2/olid_train.csv')
print('\n--- OLID TRAIN ---')
print(f'Samples: {len(olid_train)}')
print(f'Columns: {list(olid_train.columns)}')

# Distribuzione label
if 'label' in olid_train.columns:
    print('\nDistribuzione Label:')
    print(olid_train['label'].value_counts())
    
# Lunghezza testi
text_col = 'text' if 'text' in olid_train.columns else 'tweet'
olid_train['text_len'] = olid_train[text_col].astype(str).apply(len)
print('\nLunghezza Testi OLID:')
print(f'  Min: {olid_train["text_len"].min()}')
print(f'  Max: {olid_train["text_len"].max()}')
print(f'  Mean: {olid_train["text_len"].mean():.1f}')
print(f'  Median: {olid_train["text_len"].median():.1f}')

# Testi molto corti (potenziale problema)
short_texts = olid_train[olid_train['text_len'] < 20]
print(f'\nTesti molto corti (<20 char): {len(short_texts)}')
if len(short_texts) > 0:
    print('  Esempi:')
    for i, row in short_texts.head(3).iterrows():
        print(f'    "{row[text_col]}" -> Label: {row["label"]}')

# OLID Validation
olid_val = pd.read_csv('data/kaggle_upload_v2/olid_validation.csv')
print('\n--- OLID VALIDATION ---')
print(f'Samples: {len(olid_val)}')
if 'label' in olid_val.columns:
    print('\nDistribuzione Label:')
    print(olid_val['label'].value_counts())
    val_ratio = olid_val['label'].value_counts()
    print(f'\nRatio NOT:OFF = {val_ratio.get(0, val_ratio.get("NOT", 0))}:{val_ratio.get(1, val_ratio.get("OFF", 0))}')

# GoEmotions Clean
goemo = pd.read_csv('data/kaggle_upload_v2/goemotions_clean.csv')
print('\n--- GOEMOTIONS CLEAN ---')
print(f'Samples: {len(goemo)}')
emo_cols = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
print('\nDistribuzione Emozioni (samples per classe):')
for col in emo_cols:
    if col in goemo.columns:
        count = goemo[col].sum()
        pct = count / len(goemo) * 100
        bar = '#' * int(pct / 2)
        print(f'  {col:10}: {int(count):5} ({pct:5.1f}%) {bar}')

# Multi-label analysis
goemo['num_emotions'] = goemo[emo_cols].sum(axis=1)
print('\nDistribuzione Multi-Label (quante emozioni per sample):')
for n in sorted(goemo['num_emotions'].unique()):
    count = (goemo['num_emotions'] == n).sum()
    pct = count / len(goemo) * 100
    print(f'  {int(n)} emozioni: {count:5} ({pct:.1f}%)')

# Correlazione emozioni-tossicita potenziale
print('\n--- ANALISI CORRELAZIONI POTENZIALI ---')
# Simula quali emozioni potrebbero correlare con tossicita
anger_samples = goemo[goemo['anger'] == 1]['text'].head(3).tolist()
print('\nEsempi testi con ANGER (potenzialmente tossici):')
for t in anger_samples:
    print(f'  "{t[:80]}..."')

joy_samples = goemo[goemo['joy'] == 1]['text'].head(3).tolist()
print('\nEsempi testi con JOY (probabilmente non tossici):')
for t in joy_samples:
    print(f'  "{t[:80]}..."')

print('\n' + '='*60)
print('INSIGHTS')
print('='*60)
print('''
1. Se "anger" e "disgust" sono fortemente presenti in GoEmotions,
   il modello MTL dovrebbe imparare che queste emozioni correlano
   con la tossicita.
   
2. Se "joy" e "neutral" dominano, il modello potrebbe imparare
   che queste emozioni sono anti-correlate con la tossicita.
   
3. Lo sbilanciamento in OLID (2:1 NOT:OFF) e la distribuzione
   delle emozioni influenzano direttamente la capacita del modello
   di generalizzare.
''')
