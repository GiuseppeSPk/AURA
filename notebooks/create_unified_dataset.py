import pandas as pd
import os

print("🔧 Creating UNIFIED BASELINE dataset...")
print("="*60)

DATA_DIR = 'C:/Users/spicc/Desktop/Multimodal/AURA/kaggle_upload/aura-v10-data'

# Load all datasets
print("\n📂 Loading datasets...")
tox_train = pd.read_csv(f'{DATA_DIR}/toxicity_train.csv')
emo_train = pd.read_csv(f'{DATA_DIR}/emotions_train.csv')
sent_train = pd.read_csv(f'{DATA_DIR}/sentiment_train.csv')
rep_train = pd.read_csv(f'{DATA_DIR}/reporting_examples.csv')

print(f"  ✅ Toxicity:  {len(tox_train):,} samples")
print(f"  ✅ Emotion:   {len(emo_train):,} samples")
print(f"  ✅ Sentiment: {len(sent_train):,} samples")
print(f"  ✅ Reporting: {len(rep_train):,} samples")

# Prepare unified format
print("\n🔨 Standardizing format...")

# Toxicity: Keep as-is (label 0/1 for toxic)
tox_unified = pd.DataFrame({
    'text': tox_train['text'],
    'label': tox_train['label'],  # 0 = Non-Toxic, 1 = Toxic
    'task': 'toxicity'
})

# Emotion: All are non-toxic examples (label=0)
emo_unified = pd.DataFrame({
    'text': emo_train['text'],
    'label': 0,  # All emotion samples are non-toxic
    'task': 'emotion'
})

# Sentiment: All are non-toxic (label=0)
sent_unified = pd.DataFrame({
    'text': sent_train['text'],
    'label': 0,  # Sentiment doesn't indicate toxicity
    'task': 'sentiment'
})

# Reporting: All are non-toxic (label=0)
rep_unified = pd.DataFrame({
    'text': rep_train['text'],
    'label': 0,  # Reporting examples are non-toxic
    'task': 'reporting'
})

# Concatenate all
print("\n🔗 Concatenating datasets...")
unified = pd.concat([tox_unified, emo_unified, sent_unified, rep_unified], 
                    ignore_index=True)

# Shuffle for good measure
unified = unified.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = f'{DATA_DIR}/unified_baseline_train.csv'
unified.to_csv(output_path, index=False)

print(f"\n✅ Unified dataset created!")
print(f"   Total samples: {len(unified):,}")
print(f"   Saved to: unified_baseline_train.csv")

# Statistics
print(f"\n📊 Label distribution:")
print(f"   Non-Toxic (0): {(unified['label'] == 0).sum():,} ({(unified['label'] == 0).sum()/len(unified)*100:.1f}%)")
print(f"   Toxic (1):     {(unified['label'] == 1).sum():,} ({(unified['label'] == 1).sum()/len(unified)*100:.1f}%)")

print(f"\n📋 Task distribution:")
for task in ['toxicity', 'emotion', 'sentiment', 'reporting']:
    count = (unified['task'] == task).sum()
    print(f"   {task.capitalize()}: {count:,} ({count/len(unified)*100:.1f}%)")

print("\n" + "="*60)
print("🎯 READY FOR BASELINE TRAINING!")
print("="*60)
print("\nThis unified dataset can now be used for:")
print("  1. Simpler baseline notebook (single CSV load)")
print("  2. Direct comparison with professor's hypothesis")
print("  3. Clearer experimental design")
