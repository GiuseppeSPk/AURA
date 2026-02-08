import pandas as pd
import numpy as np

def check_dataset(name, path, checks):
    print(f"\n🔍 CHECKING: {name} ({path})")
    try:
        df = pd.read_csv(path)
        print(f"   Rows: {len(df)}")
        
        # Check 1: Nulls
        nulls = df.isnull().sum().sum()
        if nulls > 0:
            print(f"   ⚠️ WARNING: Found {nulls} null values!")
            print(df.isnull().sum())
        else:
            print("   ✅ No null values.")
            
        # Custom Checks
        for check_name, check_fn in checks.items():
            try:
                if check_fn(df):
                    print(f"   ✅ {check_name}: PASSED")
                else:
                    print(f"   ❌ {check_name}: FAILED")
            except Exception as e:
                print(f"   ❌ {check_name}: ERROR ({e})")
                
    except Exception as e:
        print(f"   ❌ CRITICAL: Could not read file ({e})")

# Define Checks
checks_tox = {
    "Labels are 0/1": lambda df: df['label'].isin([0, 1]).all(),
    "Text not empty": lambda df: df['text'].str.strip().str.len().min() > 0
}

checks_emo = {
    "Columns exist": lambda df: all(c in df.columns for c in ['anger', 'neutral', 'text']),
    "Values 0-1 range": lambda df: (df.iloc[:, 1:].max().max() <= 1.0) and (df.iloc[:, 1:].min().min() >= 0.0)
}

checks_rep = {
    "Labels are 0/1": lambda df: df['is_reporting'].isin([0, 1]).all(),
    "Balance check": lambda df: df['is_reporting'].mean() > 0.1 # At least 10% positives
}

# Run
check_dataset("Toxicity", "kaggle_upload/aura-v10-data/toxicity_train.csv", checks_tox)
check_dataset("Emotions", "kaggle_upload/aura-v10-data/emotions_train.csv", checks_emo)
check_dataset("Reporting", "kaggle_upload/aura-v10-data/reporting_examples.csv", checks_rep)
check_dataset("Sentiment", "kaggle_upload/aura-v10-data/sentiment_train.csv", checks_tox)
