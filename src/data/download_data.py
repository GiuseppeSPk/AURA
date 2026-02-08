import os
import argparse
from datasets import load_dataset
import pandas as pd
from preprocessing import map_goemotions_to_ekman, preprocess_text

def download_and_process():
    print("🚀 Starting Data Download & Processing...")
    
    # Create directories
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # --- 1. GoEmotions ---
    print("\n⬇️ Downloading GoEmotions (HuggingFace)...")
    try:
        # Load simplified version which already has some mapping, but raw is better
        # 'raw' config usually has all 27 emotions
        dataset = load_dataset("go_emotions", "raw") 
        df_go = dataset['train'].to_pandas()
        
        # Save raw
        df_go.to_csv(os.path.join(raw_dir, "goemotions_raw.csv"), index=False)
        print("✅ GoEmotions RAW saved.")

        # Process
        print("⚙️ Processing GoEmotions...")
        df_go = map_goemotions_to_ekman(df_go)
        df_go['text'] = df_go['text'].apply(preprocess_text)
        
        # Save processed
        output_path = os.path.join(processed_dir, "goemotions_processed.csv")
        df_go.to_csv(output_path, index=False)
        print(f"✅ GoEmotions PROCESSED saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error with GoEmotions: {e}")

    # --- 2. OLID (using tweet_eval as proxy or direct download if available) ---
    print("\n⬇️ Downloading OLID (via tweet_eval/offensive)...")
    try:
        # tweet_eval 'offensive' is based on OLID
        dataset = load_dataset("tweet_eval", "offensive")
        
        # Combine splits for our own custom splitting later if needed, 
        # or keep them. Let's save them as is.
        for split in ['train', 'validation', 'test']:
            df = dataset[split].to_pandas()
            df['text'] = df['text'].apply(preprocess_text)
            df.to_csv(os.path.join(processed_dir, f"olid_{split}.csv"), index=False)
            
        print("✅ OLID samples saved (via tweet_eval).")
        
    except Exception as e:
        print(f"❌ Error with OLID: {e}")

if __name__ == "__main__":
    download_and_process()
