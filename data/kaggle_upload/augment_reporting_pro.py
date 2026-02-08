import pandas as pd
import random

# Initial seeds
SEED = 42
random.seed(SEED)

def augment_reporting():
    print("🚀 Starting Reporting Augmentation...")
    
    # 1. Load existing reporting
    rep_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples.csv'
    df_rep = pd.read_csv(rep_path)
    print(f"   - Original Reporting: {len(df_rep)} samples")
    
    # 2. Load Tox samples for wrapping (from toxicity_train.csv)
    tox_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\toxicity_train.csv'
    df_tox = pd.read_csv(tox_path)
    toxic_texts = df_tox[df_tox['label'] == 1]['text'].tolist()
    print(f"   - Available Toxic seeds: {len(toxic_texts)} samples")
    
    # 3. Define Reporting Templates (Positive class: is_reporting=1)
    # These turn toxic content into a "reporting event", making the label NON-TOXIC for toxicity task,
    # but POSITIVE for the reporting task.
    templates_pos = [
        "According to the report, they said: '{}'",
        "The transcript shows him saying: '{}'",
        "I heard a witness shout: '{}'",
        "She claimed that he called her: '{}'",
        "The document records the insult: '{}'",
        "The victim testified that the defendant said: '{}'",
        "Someone in the crowd yelled: '{}'",
        "I am reporting a user who said: '{}'",
        "The moderator flagged this comment: '{}'",
        "He quoted the following: '{}'",
        "The evidence includes the message: '{}'",
        "They are accused of saying: '{}'",
        "The article mentions he was overheard saying: '{}'",
        "A colleague reported that she said: '{}'",
        "The screenshot captures the text: '{}'"
    ]
    
    # 4. Define Negative Templates (Negative class: is_reporting=0)
    # These look like reporting but are actually direct statements (e.g. "I say you are...")
    templates_neg = [
        "I say to you: '{}'",
        "Let me tell you: '{}'",
        "Listen to me, '{}'",
        "I'm telling you right now: '{}'",
        "I think that '{}'",
        "My opinion is: '{}'",
        "I feel that '{}'",
        "Seriously, '{}'",
        "Look, '{}'",
        "I promise you, '{}'"
    ]
    
    new_samples = []
    
    # Goal: Add ~5000 samples (2500 per class approx) to reach ~6.4k total
    num_to_add = 5000
    num_half = num_to_add // 2
    
    # Create Positives (is_reporting=1)
    for i in range(num_half):
        t = random.choice(templates_pos)
        s = random.choice(toxic_texts) # Sample directly to avoid index errors
        new_samples.append({'text': t.format(s), 'is_reporting': 1})
        
    # Create Negatives (is_reporting=0)
    for i in range(num_half):
        t = random.choice(templates_neg)
        s = random.choice(toxic_texts)
        new_samples.append({'text': t.format(s), 'is_reporting': 0})
        
    df_aug = pd.DataFrame(new_samples)
    df_final = pd.concat([df_rep, df_aug]).reset_index(drop=True)
    
    # Shuffling
    df_final = df_final.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    save_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples_augmented.csv'
    df_final.to_csv(save_path, index=False)
    print(f"✅ Success! Augmented dataset saved to: {save_path}")
    print(f"   - Final count: {len(df_final)} samples")

if __name__ == "__main__":
    augment_reporting()
