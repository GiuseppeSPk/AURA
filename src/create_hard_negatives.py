"""
AURA V12 - Hard Negatives Generator
Crea un dataset di frasi "difficili" per insegnare al modello
che RABBIA ≠ TOSSICITÀ.
"""
import pandas as pd
import os

# Hard negatives: frasi con emozioni negative ma SENZA attacco personale
HARD_NEGATIVES = [
    # ANGER (Non-Toxic)
    "I am so angry at the traffic today, it ruined my morning.",
    "I hate waking up early on Mondays.",
    "This is absolutely infuriating, the package got lost again.",
    "I'm furious that they canceled the concert.",
    "The wifi keeps disconnecting and it's driving me crazy.",
    "I can't believe they raised the prices again, so frustrating.",
    "I'm angry at myself for forgetting my keys.",
    "This weather is making me so irritable.",
    "I hate when people don't signal before turning.",
    "The customer service was terrible, I'm so annoyed.",
    "I'm fed up with these constant delays.",
    "This bugs me so much, why can't they fix it?",
    "I'm livid about the parking ticket I got.",
    "The noise from construction is unbearable.",
    "I hate how expensive everything has become.",
    
    # DISGUST (Non-Toxic)
    "This food tastes absolutely disgusting.",
    "Eww, there's mold on my bread.",
    "The smell in this room is nauseating.",
    "I find this movie repulsive, too much gore.",
    "The bathroom was filthy, I'm grossed out.",
    "This milk has gone bad, it smells awful.",
    "I can't stand the taste of liver.",
    "The texture of this dish is really off-putting.",
    "That's a disgusting habit, please stop.",
    "The garbage hasn't been collected in days, it stinks.",
    
    # SADNESS (Non-Toxic)
    "I'm devastated by the news of his passing.",
    "This breakup is killing me inside.",
    "I feel so lonely without my friends around.",
    "The ending of that movie made me cry.",
    "I'm heartbroken that we lost the game.",
    "Missing my family so much right now.",
    "I feel empty after finishing that book series.",
    "The layoffs at work have me really down.",
    "I'm so disappointed in how things turned out.",
    "This rainy day is making me feel melancholic.",
    
    # STRONG OPINIONS (Non-Toxic)
    "I completely disagree with this policy, it's a mistake.",
    "This government is failing its citizens on healthcare.",
    "The education system needs a complete overhaul.",
    "I think this movie is garbage, don't waste your time.",
    "This product is overpriced and underperforms.",
    "The new update ruined the app, terrible decision.",
    "I believe this law is unjust and should be repealed.",
    "This restaurant has gone downhill, very disappointing.",
    "The sequel was a letdown compared to the original.",
    "I strongly oppose this proposal, it's shortsighted.",
    
    # FEAR (Non-Toxic)
    "I'm terrified of flying, planes scare me.",
    "The news about the economy is really worrying.",
    "I'm anxious about the job interview tomorrow.",
    "That horror movie gave me nightmares.",
    "I'm scared of spiders, can't even look at them.",
    "The thought of public speaking makes me nervous.",
    "I'm worried about my health after those test results.",
    "This neighborhood feels unsafe at night.",
    "I'm afraid I might fail the exam.",
    "The pandemic has made me paranoid about germs.",
    
    # SARCASM (Non-Toxic but tricky)
    "Oh great, another meeting that could have been an email.",
    "Wonderful, my flight is delayed again.",
    "How lovely, it's raining on my wedding day.",
    "Fantastic, the printer jammed right before my presentation.",
    "Perfect timing, the elevator is out of order.",
    
    # COMPLAINING (Non-Toxic)
    "Why is everything so expensive nowadays?",
    "I can't believe how slow this line is moving.",
    "The service here is always so slow.",
    "Why do they keep changing things that work fine?",
    "This commute is getting worse every day.",
]

def create_hard_negatives_csv(output_path='data/aura_v9_clean/hard_negatives.csv'):
    """Create the hard negatives CSV file."""
    df = pd.DataFrame({
        'text': HARD_NEGATIVES,
        'label': [0] * len(HARD_NEGATIVES)  # All are NON-TOXIC
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Created {len(df)} hard negatives at: {output_path}")
    return df

def merge_with_toxicity_train(toxicity_path='data/aura_v9_clean/toxicity_train.csv',
                               hard_neg_path='data/aura_v9_clean/hard_negatives.csv',
                               output_path='data/aura_v9_clean/toxicity_train_v12.csv'):
    """Merge hard negatives with the original toxicity training data."""
    tox_df = pd.read_csv(toxicity_path)
    hard_df = pd.read_csv(hard_neg_path)
    
    print(f"Original toxicity train: {len(tox_df)} samples")
    print(f"Hard negatives: {len(hard_df)} samples")
    
    # Combine
    merged = pd.concat([tox_df, hard_df], ignore_index=True)
    
    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved: {len(merged)} samples at {output_path}")
    
    # Show new class distribution
    print("\nNew Class Distribution:")
    print(merged['label'].value_counts())
    
    return merged

if __name__ == "__main__":
    print("=== AURA V12 Hard Negatives Generator ===\n")
    
    # Step 1: Create hard negatives file
    create_hard_negatives_csv()
    
    # Step 2: Merge with original toxicity data
    merge_with_toxicity_train()
    
    print("\n✅ Data preparation complete. Ready for V12 training.")
