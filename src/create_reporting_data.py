"""
AURA V14 - Reporting Examples Generator
Crea un dataset per distinguere:
- Affermazioni DIRETTE ("You are stupid") → 0
- CITAZIONI/REPORT ("He said you are stupid") → 1
"""
import pandas as pd
import os

# Reporting patterns (is_reporting = 1)
REPORTING_EXAMPLES = [
    # Citazioni esplicite
    ("He said you are an idiot", 1),
    ("She called him a moron", 1),
    ("The user wrote 'go kill yourself'", 1),
    ("According to the article, hate speech is increasing", 1),
    ("The report mentions racist comments", 1),
    ("Someone posted 'I hate all of you'", 1),
    ("The tweet said 'you are worthless'", 1),
    ("In the comment, they wrote 'stupid people everywhere'", 1),
    ("The message contained the phrase 'you disgust me'", 1),
    ("He claims that she is incompetent", 1),
    ("They accused him of being a fraud", 1),
    ("The email stated 'you are fired, loser'", 1),
    ("News reports indicate violent language online", 1),
    ("The study found examples of toxic speech", 1),
    ("Witnesses say he yelled slurs", 1),
    ("The defendant allegedly said threatening words", 1),
    ("Sources claim the politician used offensive terms", 1),
    ("The document contains derogatory language", 1),
    ("Historical records show discriminatory statements", 1),
    ("The transcript reveals insulting remarks", 1),
    ("According to witnesses, he threatened violence", 1),
    ("The article discusses online harassment", 1),
    ("Research shows hate speech patterns", 1),
    ("The paper analyzes toxic language examples", 1),
    ("Media reports violent threats online", 1),
    ("Someone mentioned that she is ugly", 1),
    ("He reportedly called them terrorists", 1),
    ("The post was reported for saying 'die'", 1),
    ("Users complained about the phrase 'you suck'", 1),
    ("The banned message said 'I hope you suffer'", 1),
    ("Quoted: 'All of you are idiots'", 1),
    ("The flagged content includes 'stupid immigrants'", 1),
    ("As stated in the complaint: 'worthless human'", 1),
    ("The review mentioned 'terrible service, morons'", 1),
    ("Reported speech: 'get out of my country'", 1),
    ("The deleted tweet said 'you deserve pain'", 1),
    ("Evidence shows he wrote 'I hate you all'", 1),
    ("The removed post contained 'go back to your country'", 1),
    ("Testimony includes the phrase 'you are nothing'", 1),
    ("The archived message reads 'drop dead'", 1),
    
    # Discussioni su tossicità (meta-discourse)
    ("This study examines hate speech on social media", 1),
    ("Researchers analyzed toxic comments", 1),
    ("The platform banned users for saying slurs", 1),
    ("Content moderation flags words like 'kill'", 1),
    ("The policy addresses offensive language", 1),
    ("Training data includes examples of harassment", 1),
    ("The filter catches phrases like 'you are trash'", 1),
    ("Guidelines prohibit terms like 'retard'", 1),
    ("The dataset contains labeled toxic text", 1),
    ("Analysis of discriminatory language patterns", 1),
]

# Direct statements (is_reporting = 0)
DIRECT_EXAMPLES = [
    # Tossici diretti
    ("You are an idiot", 0),
    ("I hate you so much", 0),
    ("Go kill yourself", 0),
    ("You disgust me completely", 0),
    ("You are worthless garbage", 0),
    ("Shut up you moron", 0),
    ("I hope you suffer", 0),
    ("You are a complete loser", 0),
    ("Nobody likes you", 0),
    ("You should be ashamed", 0),
    ("You are pathetic", 0),
    ("Die already", 0),
    ("You make me sick", 0),
    ("You are the worst", 0),
    ("I despise everything about you", 0),
    ("You are a failure", 0),
    ("You deserve nothing good", 0),
    ("You are ugly inside and out", 0),
    ("Everyone hates you", 0),
    ("You are a waste of space", 0),
    ("Stupid people like you ruin everything", 0),
    ("You are an embarrassment", 0),
    ("I wish you would disappear", 0),
    ("You are absolutely useless", 0),
    ("You are a disgrace", 0),
    
    # Non tossici ma diretti
    ("I love this movie", 0),
    ("This food is delicious", 0),
    ("I am so happy today", 0),
    ("Thank you for your help", 0),
    ("This is amazing work", 0),
    ("I really appreciate it", 0),
    ("You are very kind", 0),
    ("Great job on the project", 0),
    ("I enjoy working with you", 0),
    ("This makes me smile", 0),
    
    # Negativi ma non tossici (Hard negatives per reporting)
    ("I hate rainy days", 0),
    ("This soup is terrible", 0),
    ("I am so frustrated right now", 0),
    ("This service is awful", 0),
    ("I dislike this policy", 0),
    ("The movie was boring", 0),
    ("I am angry about the delay", 0),
    ("This is really disappointing", 0),
    ("I hate when this happens", 0),
    ("The weather is miserable", 0),
    
    # Opinioni forti ma dirette
    ("The government is corrupt", 0),
    ("This law is unjust", 0),
    ("I strongly disagree", 0),
    ("This decision is wrong", 0),
    ("I think this is unfair", 0),
]

def create_reporting_dataset(output_path='data/aura_v9_clean/reporting_examples.csv'):
    """Create the reporting detection dataset."""
    all_examples = REPORTING_EXAMPLES + DIRECT_EXAMPLES
    
    df = pd.DataFrame(all_examples, columns=['text', 'is_reporting'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created reporting dataset: {len(df)} examples")
    print(f"   - Direct statements (0): {len(df[df['is_reporting']==0])}")
    print(f"   - Reports/Citations (1): {len(df[df['is_reporting']==1])}")
    print(f"   Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    print("=== AURA V14 Reporting Dataset Generator ===\n")
    df = create_reporting_dataset()
    print("\nSample entries:")
    print(df.head(10).to_string(index=False))
