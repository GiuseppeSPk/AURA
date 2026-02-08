"""
AURA V11 - Enhanced Reporting Dataset Generator
==============================================
Generates 500 new curated examples for the Reporting task with:
- Hard negatives (seem like reporting but ARE direct)
- Hard positives (implicit citations without obvious verbs)
- Diverse patterns (100+ unique templates)
- Human-like natural language variations
"""

import pandas as pd
import random
import os

# =============================================================================
# HARD NEGATIVES: Seem like reporting but ARE DIRECT (label = 0)
# These are critical for preventing false positives
# =============================================================================

HARD_NEGATIVES = [
    # First person = DIRECT, not reporting
    ("I said you're an idiot, and I mean it", 0),
    ("I told you already, you're useless", 0),
    ("I'm telling you right now: you're pathetic", 0),
    ("I just called him a moron to his face", 0),
    ("I literally wrote 'you suck' in the email", 0),
    ("I'm the one who said you're worthless", 0),
    ("I personally think you're disgusting", 0),
    ("I screamed at him: get out of here!", 0),
    
    # Collective "we" = still DIRECT
    ("We all think you're trash", 0),
    ("We told them they're incompetent", 0),
    ("We're saying this directly: you failed", 0),
    ("We called out their racist behavior", 0),
    
    # Opinion statements (not reporting facts)
    ("Everyone knows you're a fraud", 0),
    ("It's obvious that you're an idiot", 0),
    ("Clearly, you're the worst at this", 0),
    ("Obviously, they're all morons", 0),
    ("Anybody can see you're useless", 0),
    
    # Rhetorical/addressing directly
    ("Let me tell you: you're pathetic", 0),
    ("Here's the truth: you disgust me", 0),
    ("Face it: you're a failure", 0),
    ("Admit it: you're terrible at this", 0),
    ("Look, you're just not good enough", 0),
    
    # Imperative with insults
    ("Just accept that you're worthless", 0),
    ("Stop pretending you're not an idiot", 0),
    ("Don't act like you're not trash", 0),
    
    # Self-reference to past statement
    ("I already said you're disgusting, remember?", 0),
    ("Like I mentioned, you're pathetic", 0),
    ("As I was saying, you're useless", 0),
    
    # Neutral/negative direct statements (hard because contain strong words)
    ("I hate when people are like this", 0),
    ("This situation is absolutely toxic", 0),
    ("I despise this kind of attitude", 0),
    ("I find your behavior repulsive", 0),
    ("You need to stop being so aggressive", 0),
    ("Your words were incredibly hurtful", 0),
    ("That was a pathetic excuse", 0),
    ("This argument is idiotic", 0),
    ("The whole thing is garbage", 0),
    ("What a moronic decision", 0),
    
    # Questions (direct, not reporting)
    ("Why are you such an idiot?", 0),
    ("Are you really that stupid?", 0),
    ("Do you realize how pathetic that sounds?", 0),
    ("How can you be so useless?", 0),
    ("What kind of moron does that?", 0),
    
    # Conditional insults (still direct)
    ("If you weren't so stupid, you'd understand", 0),
    ("You'd know if you weren't such an idiot", 0),
    ("Anyone who isn't a moron would get this", 0),
]

# =============================================================================
# HARD POSITIVES: Implicit citations without obvious reporting verbs (label = 1)
# These teach the model to recognize subtle reporting
# =============================================================================

HARD_POSITIVES = [
    # Quotes without reporting verbs
    ("'You're an idiot' - that's what the message read", 1),
    ("'Pathetic loser' - those were the exact words", 1),
    ("'Die already' - the note said nothing else", 1),
    ("The graffiti: 'Go back where you came from'", 1),
    ("'You're worthless' - end quote", 1),
    
    # Screenshot/media references
    ("Screenshot attached: 'you're disgusting'", 1),
    ("The viral tweet: 'all of you are idiots'", 1),
    ("Posted meme caption: 'stupid people everywhere'", 1),
    ("Audio clip says: 'I hope you suffer'", 1),
    ("Video thumbnail reads: 'morons exposed'", 1),
    
    # Implied attribution
    ("Apparently, 'you're trash' is trending", 1),
    ("So the story is 'they're all morons'", 1),
    ("Word is 'you're pathetic and everyone knows'", 1),
    ("The rumor? 'You're a complete failure'", 1),
    ("There's talk of 'worthless employees'", 1),
    
    # Passive constructions
    ("The phrase 'shut up idiot' was used", 1),
    ("'You're disgusting' was written there", 1),
    ("Somewhere it says 'you're trash'", 1),
    ("'Pathetic loser' appears in the document", 1),
    ("The term 'moron' gets thrown around a lot", 1),
    
    # Meta-discussion without explicit reporting
    ("That 'you're worthless' line? Classic bullying", 1),
    ("The whole 'die already' thing is concerning", 1),
    ("Using 'idiot' as an insult is so common now", 1),
    ("'Trash' as a descriptor - very toxic", 1),
    
    # Nested/complex reporting
    ("She heard him tell you 'you're pathetic'", 1),
    ("They told me she called you 'worthless'", 1),
    ("I saw the part where it says 'morons like you'", 1),
    ("You know the bit about 'stupid people'?", 1),
    
    # Sarcastic/ironic reporting (still reporting)
    ("Oh great, another 'you're an idiot' moment", 1),
    ("Ah yes, the classic 'you're worthless' speech", 1),
    ("Lovely, they went with 'disgusting pig' again", 1),
    
    # Legal/formal implicit
    ("Exhibit A contains 'you deserve to die'", 1),
    ("Evidence item: 'pathetic excuse for a human'", 1),
    ("The contested phrase: 'worthless trash'", 1),
    ("In question: 'idiots like you'", 1),
    
    # Social media style implicit
    ("lol @ 'you're such a moron'", 1),
    ("this 'you're disgusting' energy is wild", 1),
    ("not the 'you're pathetic' discourse again", 1),
    ("'die already' trending? yikes", 1),
]

# =============================================================================
# DIVERSE REPORTING PATTERNS (100+ unique templates)
# =============================================================================

# Academic/formal patterns
ACADEMIC_PATTERNS = [
    "The study identifies '{}' as hate speech",
    "Analysis reveals prevalence of phrases like '{}'",
    "The literature documents use of '{}'",
    "Corpus data includes expressions such as '{}'",
    "The research categorizes '{}' as verbal abuse",
    "Linguistic analysis of '{}' patterns",
    "The dissertation examines '{}' usage",
    "Data shows '{}' appears frequently in toxic discourse",
    "The paper contextualizes '{}' within harassment studies",
    "Academic sources cite '{}' as discriminatory",
]

# Legal/official patterns
LEGAL_PATTERNS = [
    "Court records contain '{}'",
    "The affidavit includes the phrase '{}'",
    "Evidence submitted: '{}'",
    "Testimony quotes '{}'",
    "Legal documents reference '{}'",
    "The complaint alleges use of '{}'",
    "Case files contain '{}'",
    "The ruling mentions '{}'",
    "Deposition transcript includes '{}'",
    "The verdict references '{}'",
]

# Journalism patterns
JOURNALISM_PATTERNS = [
    "Sources confirm the statement '{}'",
    "Breaking: tweets containing '{}' go viral",
    "Reports emerge of '{}' being used",
    "Exclusive: document shows '{}'",
    "Investigation uncovers use of '{}'",
    "Inside sources reveal '{}'",
    "The leaked memo contains '{}'",
    "Whistleblower exposes '{}'",
    "Undercover footage captures '{}'",
    "Exclusive interview reveals '{}'",
]

# Social media patterns
SOCIAL_PATTERNS = [
    "the tweet literally says '{}'",
    "crying at '{}' lmaoo",
    "this person really posted '{}'",
    "not them saying '{}' unironically",
    "screen recording of '{}' going around",
    "the copypasta includes '{}'",
    "that tiktok audio says '{}'",
    "instagram story showed '{}'",
    "linkedin post of all places: '{}'",
    "discord screenshot: '{}'",
]

# Conversational/informal patterns
CONVERSATIONAL_PATTERNS = [
    "so they just said '{}' and left",
    "wait, did they actually say '{}'?",
    "you won't believe it but they wrote '{}'",
    "get this: the email said '{}'",
    "no joke, the message was '{}'",
    "apparently the text said '{}'",
    "dude they literally said '{}'",
    "I swear the comment was '{}'",
    "the voicemail? just '{}'",
    "guess what was written: '{}'",
]

# Third party attribution patterns
THIRD_PARTY_PATTERNS = [
    "According to witnesses, '{}'",
    "Bystanders heard '{}'",
    "Multiple sources confirm '{}'",
    "Those present recall '{}'",
    "Observers noted '{}'",
    "People nearby heard '{}'",
    "The crowd witnessed '{}'",
    "Staff members reported '{}'",
    "Coworkers overheard '{}'",
    "Family members mentioned '{}'",
]

# Historical/archival patterns
HISTORICAL_PATTERNS = [
    "Historical records show '{}'",
    "Archival footage contains '{}'",
    "The old letters include '{}'",
    "Documentation from that era: '{}'",
    "Period sources quote '{}'",
    "The memoir recalls '{}'",
    "Diary entries mention '{}'",
    "Correspondence reveals '{}'",
    "The autobiography states '{}'",
    "Oral histories preserve '{}'",
]

# Policy/moderation patterns
MODERATION_PATTERNS = [
    "The ban was for saying '{}'",
    "Flagged for containing '{}'",
    "Content removed: '{}'",
    "The filter caught '{}'",
    "Violations include '{}'",
    "Suspended for posting '{}'",
    "The report cites '{}'",
    "Terms violated: '{}'",
    "Removed message: '{}'",
    "Quarantined post: '{}'",
]

# =============================================================================
# TOXIC/INSULT CONTENT (to fill templates)
# =============================================================================

TOXIC_PHRASES = [
    "you're an idiot",
    "you're worthless",
    "you're pathetic",
    "you're disgusting",
    "you're a failure",
    "you're useless",
    "you're trash",
    "die already",
    "kill yourself",
    "you make me sick",
    "nobody likes you",
    "you're a loser",
    "shut up moron",
    "waste of space",
    "you're stupid",
    "go away freak",
    "you're repulsive",
    "get lost creep",
    "you're horrible",
    "you're the worst",
    "everyone hates you",
    "you deserve nothing",
    "you're embarrassing",
    "you're a joke",
    "you're disgusting garbage",
    "piece of trash",
    "absolute moron",
    "complete idiot",
    "total failure",
    "utter disgrace",
]

# =============================================================================
# DIRECT STATEMENTS (NON-REPORTING) - More variety
# =============================================================================

DIRECT_NEUTRAL = [
    ("The weather is nice today", 0),
    ("I need to finish this project", 0),
    ("Let's meet at 3pm", 0),
    ("Did you see the game last night?", 0),
    ("I'm so tired right now", 0),
    ("This coffee is amazing", 0),
    ("Traffic was horrible today", 0),
    ("I can't wait for the weekend", 0),
    ("Have you tried the new restaurant?", 0),
    ("My phone battery is dying", 0),
    ("I love this song", 0),
    ("The meeting was productive", 0),
    ("I'm learning a new skill", 0),
    ("The sunrise was beautiful", 0),
    ("I need more sleep", 0),
    ("This book is fascinating", 0),
    ("I'm thinking about vacation", 0),
    ("The food here is great", 0),
    ("I finished the report early", 0),
    ("What time is the event?", 0),
]

DIRECT_NEGATIVE_NONTOXIC = [
    ("I hate Mondays so much", 0),
    ("This traffic is killing me", 0),
    ("I'm so frustrated with this", 0),
    ("The service here is terrible", 0),
    ("I can't stand this heat", 0),
    ("This software is so buggy", 0),
    ("I'm disappointed with the results", 0),
    ("The movie was really boring", 0),
    ("I'm annoyed by the noise", 0),
    ("The price is ridiculous", 0),
    ("I disagree with that decision", 0),
    ("This policy makes no sense", 0),
    ("I'm upset about the changes", 0),
    ("The delay is unacceptable", 0),
    ("I don't like this approach", 0),
    ("The system is broken", 0),
    ("I'm tired of waiting", 0),
    ("This is a waste of time", 0),
    ("The quality has declined", 0),
    ("I'm not happy with this", 0),
]

DIRECT_TOXIC = [
    ("You're such an idiot", 0),
    ("I hate you so much", 0),
    ("You're completely worthless", 0),
    ("Shut up and go away", 0),
    ("You're pathetic", 0),
    ("Nobody wants you here", 0),
    ("You're disgusting", 0),
    ("You make me sick", 0),
    ("You're a total failure", 0),
    ("Just leave me alone creep", 0),
    ("You're the worst person", 0),
    ("I hope you suffer", 0),
    ("You deserve nothing good", 0),
    ("You're ugly inside and out", 0),
    ("Get out of my face", 0),
    ("You're an embarrassment", 0),
    ("I can't stand you", 0),
    ("You're absolutely useless", 0),
    ("You're a waste of space", 0),
    ("I despise you", 0),
]

# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_reporting_from_templates(all_patterns, phrases, count):
    """Generate reporting examples (label=1) from patterns and phrases."""
    examples = []
    for _ in range(count):
        pattern = random.choice(all_patterns)
        phrase = random.choice(phrases)
        text = pattern.format(phrase)
        examples.append((text, 1))
    return examples

def shuffle_and_dedupe(examples):
    """Remove duplicates and shuffle."""
    unique = list(set(examples))
    random.shuffle(unique)
    return unique

def generate_dataset():
    """Generate the complete enhanced dataset."""
    
    # Collect all patterns
    all_patterns = (
        ACADEMIC_PATTERNS + LEGAL_PATTERNS + JOURNALISM_PATTERNS +
        SOCIAL_PATTERNS + CONVERSATIONAL_PATTERNS + THIRD_PARTY_PATTERNS +
        HISTORICAL_PATTERNS + MODERATION_PATTERNS
    )
    
    print(f"Total unique patterns: {len(all_patterns)}")
    
    # Generate components
    generated_reporting = generate_reporting_from_templates(all_patterns, TOXIC_PHRASES, 200)
    
    # Combine all examples
    all_examples = (
        list(HARD_NEGATIVES) +  # ~50 hard negatives
        list(HARD_POSITIVES) +  # ~50 hard positives
        generated_reporting +    # 200 generated reporting
        list(DIRECT_NEUTRAL) +   # 20 neutral direct
        list(DIRECT_NEGATIVE_NONTOXIC) +  # 20 negative non-toxic
        list(DIRECT_TOXIC) +     # 20 direct toxic
        generate_reporting_from_templates(all_patterns, TOXIC_PHRASES, 140)  # Extra reporting
    )
    
    # Dedupe and shuffle
    all_examples = shuffle_and_dedupe(all_examples)
    
    # Ensure we have exactly 500
    if len(all_examples) > 500:
        all_examples = all_examples[:500]
    elif len(all_examples) < 500:
        # Generate more if needed
        extra = generate_reporting_from_templates(all_patterns, TOXIC_PHRASES, 500 - len(all_examples))
        all_examples.extend(extra)
        all_examples = shuffle_and_dedupe(all_examples)[:500]
    
    return all_examples

def main():
    print("=" * 60)
    print("AURA V11 - Enhanced Reporting Dataset Generator")
    print("=" * 60)
    
    examples = generate_dataset()
    
    # Create DataFrame
    df = pd.DataFrame(examples, columns=['text', 'is_reporting'])
    
    # Stats
    reporting_count = df['is_reporting'].sum()
    direct_count = len(df) - reporting_count
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Reporting (1): {reporting_count} ({100*reporting_count/len(df):.1f}%)")
    print(f"   Direct (0):    {direct_count} ({100*direct_count/len(df):.1f}%)")
    
    # Save
    output_path = r'C:\Users\spicc\Desktop\Multimodal\AURA\kaggle_upload\aura-v10-data\reporting_enhanced_500.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")
    
    # Show samples
    print("\n📝 Sample entries:")
    print("-" * 60)
    for _, row in df.sample(10).iterrows():
        label = "REPORT" if row['is_reporting'] == 1 else "DIRECT"
        print(f"[{label}] {row['text'][:70]}...")
    
    return df

if __name__ == "__main__":
    df = main()
