import pandas as pd
import random

# Existing domains and patterns for diversity
DOMAINS = ["Legal", "Academic", "Social Media", "News", "Professional", "Historical"]

REPORTING_PATTERNS = [
    "The document states that '{}'",
    "The author noted: '{}'",
    "They mentioned the phrase '{}'",
    "The witness testified that the defendant said '{}'",
    "In the archives, we found the message '{}'",
    "The article explicitly quotes: '{}'",
    "He claimed to have heard '{}'",
    "The transcript records the insult '{}'",
    "The user was banned for posting '{}'",
    "Sources indicate that the comment contained '{}'",
    "The study analyzed the use of the term '{}'",
    "The email subject was '{}'",
    "I was told that they called you '{}'",
    "The moderator flagged the word '{}'",
    "According to the report, the speaker used the slur '{}'",
    "The footage captures him yelling '{}'",
    "The historian cited the phrase '{}' as an example of bias.",
    "The platform's policy prohibits writing '{}'",
    "The teacher explained why '{}' is offensive.",
    "The victim remembered the attacker saying '{}'"
]

TOXIC_CONTENT = [
    "you are a total failure", "get out of here you freak", "nobody likes you", 
    "you are worthless garbage", "shut up you idiot", "you make me sick",
    "stupid people like you", "i hate your face", "you are a loser",
    "kill yourself", "die already", "you are disgusting", "ugly moron",
    "incompetent fool", "waste of oxygen", "retard", "faggot", "slut",
    "racist pig", "dumb animal", "you are pathetic", "stay away from me you creep",
    "bloody hell you are annoying", "piece of shit", "son of a bitch",
    "you are a disgrace", "cowardly snake", "disgusting pig", "brainless idiot"
]

DIRECT_GENERIC = [
    "I'm feeling very tired today.", "The weather is quite nice.", "I love learning new things.",
    "This project is coming along nicely.", "Where did I leave my keys?", "The cat is sleeping.",
    "Could you pass the salt?", "I think we should reconsider the plan.", "This is very interesting.",
    "I disagree with your point of view.", "Please leave me alone.", "I need some help here.",
    "Wait, what happened?", "The meeting starts at 10 AM.", "Have a great weekend!",
    "I don't understand the question.", "Let's go for a walk.", "This dinner is delicious.",
    "I support your decision.", "Thank you for the update."
]

def generate_new_samples(existing_df, target_total=500):
    current_count = len(existing_df)
    needed = target_total - current_count
    
    if needed <= 0:
        return existing_df
    
    half_needed = needed // 2
    
    new_data = []
    
    # Generate Reporting samples (Label 1)
    for _ in range(half_needed):
        pattern = random.choice(REPORTING_PATTERNS)
        content = random.choice(TOXIC_CONTENT)
        text = pattern.format(content)
        new_data.append({'text': text, 'is_reporting': 1})
        
    # Generate Direct samples (Label 0)
    # Mix of direct insults and direct neutral/negative non-reporting texts
    for _ in range(needed - half_needed):
        if random.random() > 0.5:
            # Direct insult
            text = random.choice(TOXIC_CONTENT).capitalize() + "."
        else:
            # Direct neutral/other
            text = random.choice(DIRECT_GENERIC)
        new_data.append({'text': text, 'is_reporting': 0})
        
    new_df = pd.DataFrame(new_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Shuffle to mix existing and new
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    
    return combined_df

if __name__ == "__main__":
    csv_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_examples.csv'
    df = pd.read_csv(csv_path)
    print(f"Current samples: {len(df)}")
    
    expanded_df = generate_new_samples(df, 500)
    print(f"New total samples: {len(expanded_df)}")
    
    # Check balance
    print(expanded_df['is_reporting'].value_counts())
    
    expanded_df.to_csv(csv_path, index=False)
    print("Dataset expanded and saved.")
