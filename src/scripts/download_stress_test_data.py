"""
AURA - Download Stress Test Datasets
Downloads ToxiGen and Jigsaw datasets for cross-domain evaluation.

ToxiGen: Machine-generated implicit hate speech (Level 3 - Hard)
Jigsaw: Wikipedia toxic comments (Level 2 - Medium)
"""

import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_toxigen():
    """
    Download ToxiGen dataset (implicit hate speech).
    Source: skg/toxigen-data on HuggingFace
    """
    print("\n" + "="*60)
    print("📥 Downloading ToxiGen (Level 3 - Implicit Hate)")
    print("="*60)
    
    try:
        # Load the annotated test set
        dataset = load_dataset("skg/toxigen-data", "annotated", split="test")
        print(f"✅ Loaded {len(dataset)} samples from ToxiGen annotated test set")
        
        # Process into our format
        processed = []
        for item in tqdm(dataset, desc="Processing ToxiGen"):
            # ToxiGen has 'generation' (text) and 'toxicity_human' or similar
            text = item.get('text', item.get('generation', ''))
            
            # Get toxicity label (varies by version)
            if 'toxicity_human' in item:
                label = 1 if item['toxicity_human'] >= 2.5 else 0  # Scale 1-5
            elif 'toxicity_ai' in item:
                label = 1 if item['toxicity_ai'] >= 0.5 else 0
            elif 'label' in item:
                label = item['label']
            else:
                # Default: check if 'prompt_label' exists
                label = 1 if item.get('prompt_label', 0) == 1 else 0
            
            target_group = item.get('target_group', item.get('group', 'unknown'))
            
            processed.append({
                'text': text.strip() if text else '',
                'label': label,
                'label_name': 'toxic' if label == 1 else 'non-toxic',
                'target_group': target_group
            })
        
        # Filter out empty texts
        processed = [p for p in processed if p['text']]
        
        # Save as JSON
        output_path = OUTPUT_DIR / "toxigen_test.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)
        
        # Stats
        toxic_count = sum(1 for p in processed if p['label'] == 1)
        print(f"✅ Saved {len(processed)} samples to {output_path}")
        print(f"   Toxic: {toxic_count} ({100*toxic_count/len(processed):.1f}%)")
        print(f"   Non-toxic: {len(processed)-toxic_count} ({100*(len(processed)-toxic_count)/len(processed):.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading ToxiGen: {e}")
        print("   Trying alternative source...")
        
        try:
            # Alternative: try the smaller prompts version
            dataset = load_dataset("skg/toxigen-data", "prompts", split="train[:10000]")
            print(f"✅ Loaded {len(dataset)} samples from ToxiGen prompts")
            
            processed = []
            for item in tqdm(dataset, desc="Processing ToxiGen (prompts)"):
                text = item.get('text', item.get('prompt', ''))
                label = item.get('target_label', 0)
                
                processed.append({
                    'text': text.strip() if text else '',
                    'label': 1 if label == 1 else 0,
                    'label_name': 'toxic' if label == 1 else 'non-toxic',
                    'target_group': item.get('target_group', 'unknown')
                })
            
            processed = [p for p in processed if p['text']]
            
            output_path = OUTPUT_DIR / "toxigen_test.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(processed)} samples to {output_path}")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative also failed: {e2}")
            return False


def download_jigsaw():
    """
    Download Jigsaw Toxic Comment dataset (Wikipedia comments).
    Source: google/civil_comments on HuggingFace (similar domain)
    
    Note: Original Jigsaw is on Kaggle and requires authentication.
    We use civil_comments as a proxy (same Wikipedia-style comments).
    """
    print("\n" + "="*60)
    print("📥 Downloading Jigsaw/Civil Comments (Level 2 - Wikipedia Domain)")
    print("="*60)
    
    try:
        # Civil Comments is similar to Jigsaw (Wikipedia-style comments)
        dataset = load_dataset("google/civil_comments", split="test[:10000]")
        print(f"✅ Loaded {len(dataset)} samples from Civil Comments")
        
        processed = []
        for item in tqdm(dataset, desc="Processing Civil Comments"):
            text = item.get('text', '')
            
            # toxicity score 0-1, threshold at 0.5
            toxicity = item.get('toxicity', 0)
            label = 1 if toxicity >= 0.5 else 0
            
            processed.append({
                'text': text.strip() if text else '',
                'label': label,
                'label_name': 'toxic' if label == 1 else 'non-toxic',
                'toxicity_score': toxicity
            })
        
        processed = [p for p in processed if p['text']]
        
        # Balance the dataset (50/50)
        toxic = [p for p in processed if p['label'] == 1]
        non_toxic = [p for p in processed if p['label'] == 0]
        
        min_count = min(len(toxic), len(non_toxic), 5000)
        balanced = toxic[:min_count] + non_toxic[:min_count]
        
        # Shuffle
        import random
        random.shuffle(balanced)
        
        output_path = OUTPUT_DIR / "jigsaw_test.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(balanced, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved {len(balanced)} samples to {output_path}")
        print(f"   Balanced: {min_count} toxic + {min_count} non-toxic")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading Civil Comments: {e}")
        return False


def copy_class_weights():
    """Copy class weights from 2P_Work to data/processed"""
    print("\n" + "="*60)
    print("📋 Copying class weights")
    print("="*60)
    
    source = Path(__file__).parent.parent.parent / "Module_3" / "2P_Work" / "class_weights.json"
    dest = OUTPUT_DIR / "class_weights.json"
    
    if source.exists():
        import shutil
        shutil.copy(source, dest)
        print(f"✅ Copied class_weights.json to {dest}")
        return True
    else:
        print(f"⚠️ Source not found: {source}")
        return False


def main():
    print("🚀 AURA - Downloading Stress Test Datasets")
    print("="*60)
    
    success_toxigen = download_toxigen()
    success_jigsaw = download_jigsaw()
    success_weights = copy_class_weights()
    
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    print(f"ToxiGen (Level 3):    {'✅ Success' if success_toxigen else '❌ Failed'}")
    print(f"Jigsaw (Level 2):     {'✅ Success' if success_jigsaw else '❌ Failed'}")
    print(f"Class Weights:        {'✅ Success' if success_weights else '❌ Failed'}")
    
    if success_toxigen and success_jigsaw:
        print("\n🎉 All datasets ready! You can now run the 3-tier stress test.")
    else:
        print("\n⚠️ Some downloads failed. Check errors above.")


if __name__ == "__main__":
    main()
