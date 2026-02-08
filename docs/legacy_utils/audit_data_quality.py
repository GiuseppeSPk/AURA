import pandas as pd
import numpy as np

print('='*60)
print('DEEP DATA QUALITY AUDIT')
print('='*60)

df = pd.read_csv('data/kaggle_mega/toxicity_train.csv')

# 1. Text length analysis
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
print('\n[1] TEXT LENGTH ANALYSIS')
print(f'  Mean words: {df["word_count"].mean():.1f}')
print(f'  Median: {df["word_count"].median():.0f}')
print(f'  Min: {df["word_count"].min()}')
print(f'  Max: {df["word_count"].max()}')
print(f'  <5 words: {(df["word_count"] < 5).sum()} ({(df["word_count"] < 5).mean()*100:.1f}%)')
print(f'  <10 words: {(df["word_count"] < 10).sum()} ({(df["word_count"] < 10).mean()*100:.1f}%)')

# 2. Label distribution
print('\n[2] LABEL DISTRIBUTION')
print(df['label'].value_counts())
print(f'  Imbalance ratio: {df["label"].value_counts()[0] / df["label"].value_counts()[1]:.2f}:1')

# 3. Potential duplicates
print('\n[3] DUPLICATE ANALYSIS')
exact_dups = df.duplicated(subset=['text']).sum()
print(f'  Exact duplicates: {exact_dups}')

# 4. Common patterns in toxic vs non-toxic
print('\n[4] VOCABULARY ANALYSIS')
toxic = df[df['label'] == 1]['text'].str.lower().str.cat(sep=' ')
non_toxic = df[df['label'] == 0]['text'].str.lower().str.cat(sep=' ')

from collections import Counter
toxic_words = Counter(toxic.split()).most_common(10)
non_toxic_words = Counter(non_toxic.split()).most_common(10)

print('  Top Toxic:', [w[0] for w in toxic_words])
print('  Top Non-Toxic:', [w[0] for w in non_toxic_words])

# 5. Ambiguous cases
print('\n[5] EXAMPLE SHORT TWEETS (potentially ambiguous)')
short = df[df['word_count'] <= 5].sample(min(5, len(df[df['word_count'] <= 5])), random_state=42)
for _, row in short.iterrows():
    snippet = row["text"][:60]
    print(f'  [{row["label"]}] "{snippet}"')
