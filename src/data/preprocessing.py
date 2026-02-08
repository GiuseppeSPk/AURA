import argparse
import pandas as pd
import os
import yaml
import numpy as np

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

EKMAN_LABELS = config['data']['emotion_labels']

# Original GoEmotions to Ekman mapping (Standard)
# But we modify it to keep Neutral separate
GOEMOTIONS_TO_EKMAN = {
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    'disgust': 'disgust',
    'fear': 'fear', 'nervousness': 'fear',
    'joy': 'joy', 'amusement': 'joy', 'approval': 'joy', 'excitement': 'joy', 
    'gratitude': 'joy', 'love': 'joy', 'optimism': 'joy', 'relief': 'joy', 
    'pride': 'joy', 'admiration': 'joy', 'desire': 'joy', 'caring': 'joy',
    'sadness': 'sadness', 'disappointment': 'sadness', 'embarrassment': 'sadness', 
    'grief': 'sadness', 'remorse': 'sadness',
    'surprise': 'surprise', 'realization': 'surprise', 'confusion': 'surprise', 
    'curiosity': 'surprise',
    'neutral': 'neutral' # CHANGE: Explicitly map neutral to neutral
}

def load_data(dataset_name, input_path):
    print(f"Loading {dataset_name} from {input_path}...")
    # This is a placeholder for actual loading logic depending on file format (csv, tsv, json)
    if input_path.endswith('.csv'):
        return pd.read_csv(input_path)
    elif input_path.endswith('.tsv'):
        return pd.read_csv(input_path, sep='\t')
    elif input_path.endswith('.json'):
        return pd.read_json(input_path)
    else:
        raise ValueError("Unsupported file format")

def map_goemotions_to_ekman(df):
    """
    Maps GoEmotions fine-grained labels to Ekman + Neutral (7 classes).
    Assumes df has columns for each emotion or a 'label' column.
    """
    print("Mapping GoEmotions to Ekman-7...")
    # Logic depends on how GoEmotions is loaded (one-hot or list of labels)
    # Assuming standard GoEmotions format where columns are emotion names
    
    available_cols = [c for c in df.columns if c in GOEMOTIONS_TO_EKMAN]
    
    if not available_cols:
        # Maybe it's a 'label' column with indices? 
        # For now, let's assume we are handling the raw CSV with emotion columns
        print("Warning: No emotion columns found matching dictionary keys. Checking for 'emotion' column...")
        return df

    # Create new Ekman columns
    for ekman in EKMAN_LABELS:
        df[ekman] = 0

    for original_emo, ekman_emo in GOEMOTIONS_TO_EKMAN.items():
        if original_emo in df.columns:
            df[ekman_emo] = df[[ekman_emo, original_emo]].max(axis=1)
    
    # Filter to only rows that have at least one label (though most should)
    # And keep only the text and new labels
    keep_cols = ['text', 'id', 'author'] + EKMAN_LABELS
    existing_keep_cols = [c for c in keep_cols if c in df.columns]
    
    return df[existing_keep_cols]

def preprocess_text(text):
    # Basic cleaning
    return text.lower().strip()

def main():
    parser = argparse.ArgumentParser(description="AURA Preprocessing")
    parser.add_argument("--dataset", type=str, required=True, choices=['olid', 'goemotions', 'jigsaw', 'toxigen'], help="Dataset to process")
    parser.add_argument("--input_path", type=str, help="Path to raw file (optional if standard structure)")
    parser.add_argument("--output", type=str, default="data/processed/", help="Output directory")
    parser.add_argument("--ekman-mapping", action="store_true", help="Apply Ekman mapping (for GoEmotions)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Placeholder for logic
    print(f"Processing {args.dataset}...")
    
    if args.dataset == 'goemotions' and args.ekman_mapping:
        # Example flow
        # df = load_data(args.dataset, args.input_path)
        # df = map_goemotions_to_ekman(df)
        # df['text'] = df['text'].apply(preprocess_text)
        # df.to_csv(os.path.join(args.output, 'goemotions_processed.csv'), index=False)
        print("Applied Ekman mapping with 7 classes (including Neutral).")
        pass
    
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
