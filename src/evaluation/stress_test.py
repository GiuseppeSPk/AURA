"""
AURA - 3-Tier Stress Test Evaluation
=====================================
Evaluates trained model on:
- Level 1: OLID test (in-domain, Twitter)
- Level 2: Jigsaw/Civil Comments (domain shift, Wikipedia)
- Level 3: ToxiGen (implicit hate, machine-generated)

Calculates ΔF1 to measure cross-domain robustness.
"""

import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from torch.cuda.amp import autocast
except ImportError:
    from torch.amp import autocast

from src.models.aura import AURA


class StressTestDataset(Dataset):
    """Dataset for stress test evaluation (JSON format)."""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            df = pd.read_csv(data_path)
            self.data = df.to_dict('records')
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item.get('text', ''))
        label = item.get('label', 0)
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def evaluate_on_dataset(model, loader, device, dataset_name):
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            texts = batch['text']
            
            with autocast(device_type='cuda', dtype=torch.float16):
                tox_logits, _ = model(input_ids, attention_mask)
            
            probs = torch.softmax(tox_logits.float(), dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_texts.extend(texts)
    
    # Calculate metrics
    f1_macro = f1_score(all_true, all_preds, average='macro')
    f1_weighted = f1_score(all_true, all_preds, average='weighted')
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_true, all_preds, average='macro', zero_division=0)
    report = classification_report(all_true, all_preds, target_names=['non-toxic', 'toxic'], output_dict=True)
    cm = confusion_matrix(all_true, all_preds)
    
    # Error analysis
    errors = []
    for i, (pred, true, text, prob) in enumerate(zip(all_preds, all_true, all_texts, all_probs)):
        if pred != true:
            errors.append({
                'text': text[:200],  # Truncate for display
                'true_label': 'toxic' if true == 1 else 'non-toxic',
                'pred_label': 'toxic' if pred == 1 else 'non-toxic',
                'confidence': float(prob) if pred == 1 else float(1 - prob),
                'error_type': 'FP' if pred == 1 and true == 0 else 'FN'
            })
    
    return {
        'dataset': dataset_name,
        'samples': len(all_true),
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'errors': errors[:20]  # Top 20 errors
    }


def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="AURA 3-Tier Stress Test")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/aura_best.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 AURA - 3-Tier Stress Test Evaluation")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = AURA(args.config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model.eval()
    
    # Tokenizer
    with open(args.config, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder'])
    
    # Define test datasets
    test_levels = [
        {
            'level': 1,
            'name': 'OLID (In-Domain)',
            'path': 'data/processed/olid_test.csv',
            'difficulty': 'Easy'
        },
        {
            'level': 2,
            'name': 'Jigsaw (Wikipedia)',
            'path': 'data/processed/jigsaw_test.json',
            'difficulty': 'Medium'
        },
        {
            'level': 3,
            'name': 'ToxiGen (Implicit)',
            'path': 'data/processed/toxigen_test.json',
            'difficulty': 'Hard'
        }
    ]
    
    results = []
    os.makedirs("outputs/figures", exist_ok=True)
    
    for test in test_levels:
        if not os.path.exists(test['path']):
            print(f"\n⚠️ Skipping Level {test['level']}: {test['path']} not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"📊 Level {test['level']}: {test['name']} ({test['difficulty']})")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = StressTestDataset(test['path'], tokenizer, config['data']['max_length'])
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        
        # Evaluate
        result = evaluate_on_dataset(model, loader, device, test['name'])
        result['level'] = test['level']
        result['difficulty'] = test['difficulty']
        results.append(result)
        
        # Print results
        print(f"\n📈 Results:")
        print(f"   Samples: {result['samples']}")
        print(f"   Macro-F1: {result['f1_macro']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        
        # Plot confusion matrix
        cm = np.array(result['confusion_matrix'])
        plot_confusion_matrix(
            cm, 
            ['non-toxic', 'toxic'],
            f"Level {test['level']}: {test['name']}",
            f"outputs/figures/confusion_matrix_level{test['level']}.png"
        )
        print(f"   📊 Confusion matrix saved to outputs/figures/")
        
        # Print sample errors
        if result['errors']:
            print(f"\n   🔍 Sample Errors ({len(result['errors'])} shown):")
            for i, err in enumerate(result['errors'][:5]):
                print(f"      [{err['error_type']}] True: {err['true_label']}, Pred: {err['pred_label']}")
                print(f"          Text: {err['text'][:80]}...")
    
    # Summary
    print("\n" + "="*60)
    print("📊 STRESS TEST SUMMARY")
    print("="*60)
    
    if len(results) >= 1:
        print(f"\n{'Level':<8} {'Dataset':<25} {'Macro-F1':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*70)
        for r in results:
            print(f"{r['level']:<8} {r['dataset']:<25} {r['f1_macro']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")
    
    # Calculate ΔF1
    if len(results) >= 2:
        delta_f1_1_to_last = results[0]['f1_macro'] - results[-1]['f1_macro']
        print(f"\n🎯 ΔF1 (Level 1 → Level {results[-1]['level']}): {delta_f1_1_to_last:.4f}")
        
        if delta_f1_1_to_last < 0.15:
            print("   ✅ Good cross-domain robustness!")
        elif delta_f1_1_to_last < 0.25:
            print("   ⚠️ Moderate performance drop")
        else:
            print("   ❌ Significant domain shift problem")
    
    # Save full results
    output_file = "outputs/stress_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = []
        for r in results:
            jr = {k: v for k, v in r.items() if k != 'errors'}
            jr['sample_errors'] = r['errors'][:10]  # Keep only 10 errors
            json_results.append(jr)
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Full results saved to {output_file}")


if __name__ == "__main__":
    main()
