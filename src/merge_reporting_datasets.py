"""
AURA V11 - Final Reporting Dataset Merger
==========================================
Combines original 500 samples with 500 new enhanced samples.
Ensures perfect 50/50 balance between Direct (0) and Reporting (1).
"""

import pandas as pd
import random

def balance_and_merge():
    # Load datasets
    original_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples.csv'
    enhanced_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_enhanced_500.csv'
    
    original_df = pd.read_csv(original_path)
    enhanced_df = pd.read_csv(enhanced_path)
    
    print("=" * 60)
    print("ORIGINAL DATASET")
    print(f"  Total: {len(original_df)}")
    print(original_df['is_reporting'].value_counts())
    
    print("\nENHANCED DATASET")
    print(f"  Total: {len(enhanced_df)}")
    print(enhanced_df['is_reporting'].value_counts())
    
    # Combine both
    combined = pd.concat([original_df, enhanced_df], ignore_index=True)
    
    # Remove exact duplicates
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    
    print("\nCOMBINED (after dedup)")
    print(f"  Total: {len(combined)}")
    print(combined['is_reporting'].value_counts())
    
    # Split by class
    reporting = combined[combined['is_reporting'] == 1].copy()
    direct = combined[combined['is_reporting'] == 0].copy()
    
    print(f"\nAvailable: {len(reporting)} reporting, {len(direct)} direct")
    
    # Target: 500 reporting + 500 direct = 1000 balanced
    target_per_class = 500
    
    # Sample to balance
    if len(reporting) >= target_per_class:
        reporting_final = reporting.sample(n=target_per_class, random_state=42)
    else:
        reporting_final = reporting  # Use all if not enough
        print(f"WARNING: Only {len(reporting)} reporting samples available")
    
    if len(direct) >= target_per_class:
        direct_final = direct.sample(n=target_per_class, random_state=42)
    else:
        direct_final = direct  # Use all if not enough
        print(f"WARNING: Only {len(direct)} direct samples available")
    
    # Combine final balanced dataset
    final_df = pd.concat([reporting_final, direct_final], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print("\n" + "=" * 60)
    print("FINAL BALANCED DATASET")
    print("=" * 60)
    print(f"  Total: {len(final_df)}")
    print(final_df['is_reporting'].value_counts())
    
    # Create train/val split (90/10)
    train_size = int(len(final_df) * 0.9)
    train_df = final_df.iloc[:train_size]
    val_df = final_df.iloc[train_size:]
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}")
    print("Train distribution:")
    print(train_df['is_reporting'].value_counts())
    print("Val distribution:")
    print(val_df['is_reporting'].value_counts())
    
    # Save
    train_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_train.csv'
    val_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_validation.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Saved train: {train_path}")
    print(f"✅ Saved val: {val_path}")
    
    # Show samples
    print("\n📝 Sample DIRECT (0):")
    for text in direct_final.sample(5)['text'].tolist():
        print(f"  • {text[:80]}")
    
    print("\n📝 Sample REPORTING (1):")
    for text in reporting_final.sample(5)['text'].tolist():
        print(f"  • {text[:80]}")
    
    return train_df, val_df

if __name__ == "__main__":
    train_df, val_df = balance_and_merge()
