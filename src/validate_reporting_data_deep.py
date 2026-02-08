import pandas as pd
import re

def validate_dataset(path):
    print(f"🔍 Starting deep validation for: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    issues = []
    
    # 1. Basic Structure
    if list(df.columns) != ['text', 'is_reporting']:
        issues.append(f"Wrong column names: {df.columns.tolist()}")

    # 2. General Stats
    print(f"   Total rows: {len(df)}")
    print(f"   Label distribution:\n{df['is_reporting'].value_counts()}")

    # 3. Missing Values
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        issues.append(f"Found {nulls} missing values.")

    # 4. Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate rows.")
        # Optional: showing some duplicates
        # print(df[df.duplicated(keep=False)].head())

    # 5. Label Integrity
    invalid_labels = df[~df['is_reporting'].isin([0, 1])]
    if len(invalid_labels) > 0:
        issues.append(f"Found {len(invalid_labels)} invalid labels.")

    # 6. Text Quality
    empty_texts = df[df['text'].str.strip() == ""]
    if len(empty_texts) > 0:
        issues.append(f"Found {len(empty_texts)} empty text strings.")

    # 7. Semantic Validation (Reporting Label 1)
    # Check if Label 1 rows contain reporting keywords
    reporting_keywords = ['said', 'claim', 'state', 'note', 'mention', 'quote', 'cite', 'report', 'wrote', 'stated', 'testified', 'recorded', 'found']
    label_1_df = df[df['is_reporting'] == 1]
    bad_reporting = []
    for idx, text in label_1_df['text'].items():
        if not any(kw in text.lower() for kw in reporting_keywords):
            bad_reporting.append((idx, text))
    
    if len(bad_reporting) > 0:
        issues.append(f"Found {len(bad_reporting)} Reporting samples (Label 1) missing keywords.")
        print("\n⚠️ Samples Label 1 without keywords:")
        for idx, text in bad_reporting[:5]:
            print(f"   [{idx}] {text}")

    # 8. Semantic Validation (Direct Label 0)
    # Check if Label 0 rows erroneously contain reporting patterns
    label_0_df = df[df['is_reporting'] == 0]
    false_positives = []
    # Strict patterns that usually imply reporting
    reporting_patterns = [r"he said", r"she said", r"they said", r"it states", r"the report", r"quoted:", r"claimed that"]
    for idx, text in label_0_df['text'].items():
        if any(re.search(p, text.lower()) for p in reporting_patterns):
            false_positives.append((idx, text))
    
    if len(false_positives) > 0:
        issues.append(f"Found {len(false_positives)} Direct samples (Label 0) with strong reporting patterns.")
        print("\n⚠️ Samples Label 0 with reporting patterns:")
        for idx, text in false_positives[:5]:
            print(f"   [{idx}] {text}")

    # Final Report
    print("\n" + "="*40)
    if not issues:
        print("✅ VALIDATION SUCCESSFUL: No issues found.")
    else:
        print(f"❌ VALIDATION FAILED: Found {len(issues)} issues.")
        for issue in issues:
            print(f"   - {issue}")
    print("="*40)

if __name__ == "__main__":
    path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples.csv'
    validate_dataset(path)
