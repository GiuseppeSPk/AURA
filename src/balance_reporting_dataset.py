"""
AURA V11 - Additional Direct Examples Generator
================================================
Generates more label=0 (direct) examples to balance the dataset.
Focus on hard negatives and diverse direct statements.
"""

import pandas as pd
import random

# =============================================================================
# ADDITIONAL HARD NEGATIVES (label=0)
# =============================================================================

ADDITIONAL_HARD_NEGATIVES = [
    # First person statements (I/We = always direct)
    ("I called him an idiot yesterday", 0),
    ("I'm saying this clearly: you're worthless", 0),
    ("I wrote that email saying you're trash", 0),
    ("I yelled at them for being stupid", 0),
    ("I posted that comment about morons", 0),
    ("I tweeted that you're disgusting", 0),
    ("I texted him saying he's pathetic", 0),
    ("I replied calling her useless", 0),
    ("I commented that they're all losers", 0),
    ("I messaged saying you're a failure", 0),
    ("We're calling them out as frauds", 0),
    ("We posted about their stupidity", 0),
    ("We wrote that scathing review", 0),
    ("We told everyone you're garbage", 0),
    ("We announced they're incompetent", 0),
    
    # Opinions presented as facts
    ("It's well known that you're stupid", 0),
    ("Everyone agrees you're pathetic", 0),
    ("Clearly you're the problem here", 0),
    ("Obviously you're an idiot", 0),
    ("Undoubtedly they're all failures", 0),
    ("Certainly you're worthless", 0),
    ("Definitely you're trash", 0),
    ("Surely you're disgusting", 0),
    ("Of course you're useless", 0),
    ("Naturally they're morons", 0),
    
    # Commands and imperatives
    ("Admit you're an idiot", 0),
    ("Accept that you're worthless", 0),
    ("Realize you're pathetic", 0),
    ("Understand you're trash", 0),
    ("Acknowledge you're useless", 0),
    ("Face that you're a failure", 0),
    ("Own up to being stupid", 0),
    ("Confess you're disgusting", 0),
    
    # Questions as insults
    ("Why are you so stupid?", 0),
    ("How are you this pathetic?", 0),
    ("When will you stop being trash?", 0),
    ("What makes you such a loser?", 0),
    ("Who raised you to be this useless?", 0),
    ("Where did you learn to be so dumb?", 0),
    ("Can't you see you're worthless?", 0),
    ("Don't you know you're pathetic?", 0),
    ("Won't you admit you're stupid?", 0),
    ("Isn't it obvious you're garbage?", 0),
    
    # "Let me" and "Here's" patterns
    ("Let me be clear: you're an idiot", 0),
    ("Let me say this: you're trash", 0),
    ("Let me explain: you're worthless", 0),
    ("Here's the thing: you're pathetic", 0),
    ("Here's what I think: you're stupid", 0),
    ("Here's my take: you're useless", 0),
    ("Here's the truth: you're garbage", 0),
    
    # Comparative insults
    ("You're worse than useless", 0),
    ("You're more pathetic than I thought", 0),
    ("You're even stupider than expected", 0),
    ("You're the worst person I know", 0),
    ("You're more worthless than trash", 0),
    ("You're dumber than a rock", 0),
    
    # Time-referenced insults (still direct)
    ("You've always been an idiot", 0),
    ("You'll always be worthless", 0),
    ("You've never been anything but trash", 0),
    ("You're still pathetic after all this time", 0),
    ("You remain as useless as ever", 0),
    
    # Conditional insults
    ("Even if you tried, you're still stupid", 0),
    ("No matter what, you're pathetic", 0),
    ("Despite everything, you're useless", 0),
    ("Regardless, you're trash", 0),
    ("Either way, you're an idiot", 0),
    
    # Exclamatory insults
    ("What an idiot you are!", 0),
    ("How pathetic!", 0),
    ("Such a loser!", 0),
    ("Total garbage!", 0),
    ("Absolute moron!", 0),
    ("Complete waste of space!", 0),
    ("Utter disgrace!", 0),
    ("Pure trash!", 0),
]

# =============================================================================
# ADDITIONAL NEUTRAL/NEGATIVE DIRECT (label=0)
# =============================================================================

ADDITIONAL_DIRECT = [
    # Workplace/professional
    ("This project is behind schedule", 0),
    ("The deadline was missed again", 0),
    ("Performance has been declining", 0),
    ("We need to improve productivity", 0),
    ("The report has several errors", 0),
    ("This requires significant revision", 0),
    ("The proposal was rejected", 0),
    ("Budget constraints are tight", 0),
    
    # Daily life observations
    ("The train is late again", 0),
    ("My internet connection is terrible", 0),
    ("The queue is way too long", 0),
    ("This product is overpriced", 0),
    ("The service was disappointing", 0),
    ("My order was wrong again", 0),
    ("The parking situation is awful", 0),
    ("This neighborhood is noisy", 0),
    
    # Opinions and preferences
    ("I don't agree with this policy", 0),
    ("This approach seems flawed", 0),
    ("I have concerns about this plan", 0),
    ("This doesn't make sense to me", 0),
    ("I'm skeptical about the results", 0),
    ("This needs more consideration", 0),
    ("I question this methodology", 0),
    ("This argument is unconvincing", 0),
    
    # Frustrations (non-toxic)
    ("Why is this so complicated?", 0),
    ("This keeps happening to me", 0),
    ("I can't figure this out", 0),
    ("Nothing is working today", 0),
    ("Everything is going wrong", 0),
    ("I'm having the worst luck", 0),
    ("This day has been terrible", 0),
    ("I need a break from this", 0),
    
    # Casual conversation
    ("Did you watch the match?", 0),
    ("What are you doing this weekend?", 0),
    ("Have you tried the new place?", 0),
    ("When is the next meeting?", 0),
    ("Who's joining us for lunch?", 0),
    ("Where should we go?", 0),
    ("How was your trip?", 0),
    ("What do you think about this?", 0),
    
    # Positive statements
    ("This turned out great", 0),
    ("I'm really happy with the results", 0),
    ("Everything is going well", 0),
    ("This is exactly what I wanted", 0),
    ("The team did an excellent job", 0),
    ("I appreciate your help", 0),
    ("Thanks for your support", 0),
    ("Great work everyone", 0),
]

def generate_additional_direct():
    """Generate additional direct examples."""
    all_examples = ADDITIONAL_HARD_NEGATIVES + ADDITIONAL_DIRECT
    
    # Remove duplicates
    all_examples = list(set(all_examples))
    random.shuffle(all_examples)
    
    print(f"Generated {len(all_examples)} additional direct examples")
    return all_examples

def merge_and_balance():
    """Merge all datasets and create balanced train/val splits."""
    
    # Load existing datasets
    original_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples.csv'
    enhanced_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_enhanced_500.csv'
    
    original_df = pd.read_csv(original_path)
    enhanced_df = pd.read_csv(enhanced_path)
    
    # Generate additional direct examples
    additional_direct = generate_additional_direct()
    additional_df = pd.DataFrame(additional_direct, columns=['text', 'is_reporting'])
    
    # Combine all
    combined = pd.concat([original_df, enhanced_df, additional_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'], keep='first')
    
    print(f"\nTotal after merge and dedup: {len(combined)}")
    print(combined['is_reporting'].value_counts())
    
    # Split by class
    reporting = combined[combined['is_reporting'] == 1]
    direct = combined[combined['is_reporting'] == 0]
    
    print(f"\nAvailable: {len(reporting)} reporting, {len(direct)} direct")
    
    # Take equal amounts from each class
    min_size = min(len(reporting), len(direct))
    target_per_class = min(min_size, 500)  # Cap at 500 per class
    
    print(f"Sampling {target_per_class} per class for balance")
    
    reporting_sampled = reporting.sample(n=target_per_class, random_state=42)
    direct_sampled = direct.sample(n=target_per_class, random_state=42)
    
    # Combine and shuffle
    final_df = pd.concat([reporting_sampled, direct_sampled], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal balanced: {len(final_df)}")
    print(final_df['is_reporting'].value_counts())
    
    # Train/val split (90/10)
    train_size = int(len(final_df) * 0.9)
    train_df = final_df.iloc[:train_size]
    val_df = final_df.iloc[train_size:]
    
    print(f"\nTrain: {len(train_df)}")
    print(train_df['is_reporting'].value_counts())
    print(f"\nVal: {len(val_df)}")
    print(val_df['is_reporting'].value_counts())
    
    # Save
    train_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_train.csv'
    val_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_validation.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✅ Saved train: {train_path} ({len(train_df)} samples)")
    print(f"✅ Saved val: {val_path} ({len(val_df)} samples)")
    
    return train_df, val_df

if __name__ == "__main__":
    print("=" * 60)
    print("AURA V11 - Balanced Reporting Dataset Generator")
    print("=" * 60)
    train_df, val_df = merge_and_balance()
