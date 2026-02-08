import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report

from src.models.aura import AURA
from src.data.aura_dataset import AURADataset
from src.models.losses import UncertaintyLoss, FocalLoss

def train_one_epoch(model, loader, optimizer, uncertainty_loss, device, epoch):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    
    for batch in loop:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tox_targets = batch['toxicity_target'].to(device)
        emo_targets = batch['emotion_target'].to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        tox_logits, emo_logits = model(input_ids, attention_mask)
        
        # Loss Calculation
        # 1. Toxicity Loss (CrossEntropy - Ignore -1 labels if any)
        # Note: OLID has 0/1. If dataset mixes in GoEmotions, tox might be -1.
        # We need to mask out -1s.
        tox_mask = tox_targets != -1
        if tox_mask.sum() > 0:
            loss_t = nn.CrossEntropyLoss()(tox_logits[tox_mask], tox_targets[tox_mask])
        else:
            loss_t = torch.tensor(0.0, device=device, requires_grad=True)

        # 2. Emotion Loss (BCEWithLogitsLoss - Ignore -1 labels)
        # Assuming emo_targets are floats (multi-hot or smoothed)
        # If -1 in targets (e.g. from OLID), mask them.
        # We check the first column to see if it's valid
        emo_mask = emo_targets[:, 0] != -1
        if emo_mask.sum() > 0:
            loss_e = nn.BCEWithLogitsLoss()(emo_logits[emo_mask], emo_targets[emo_mask])
        else:
            loss_e = torch.tensor(0.0, device=device, requires_grad=True)
            
        # 3. Combined Loss (Uncertainty Weighting)
        loss = uncertainty_loss(loss_t, loss_e)
        
        # Backward
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item(), loss_t=loss_t.item(), loss_e=loss_e.item())
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_tox_preds = []
    all_tox_true = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tox_targets = batch['toxicity_target'].to(device)
            
            tox_logits, _ = model(input_ids, attention_mask)
            
            # Predict Toxicity
            probs = torch.softmax(tox_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Filter valid (not -1)
            mask = tox_targets != -1
            if mask.sum() > 0:
                all_tox_preds.extend(preds[mask].cpu().numpy())
                all_tox_true.extend(tox_targets[mask].cpu().numpy())
                
    f1 = f1_score(all_tox_true, all_tox_preds, average='macro')
    report = classification_report(all_tox_true, all_tox_preds, output_dict=True)
    return f1, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default="aura")
    args = parser.parse_args()
    
    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    
    # Device Selection
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"Using device: {device} (Intel Arc/DirectML)")
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: {device} (CUDA)")
        else:
            device = torch.device("cpu")
            print(f"Using device: {device} (CPU)")
    
    # Model
    model = AURA(args.config).to(device)
    uncertainty_loss = UncertaintyLoss().to(device)
    
    # Optimizer
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': float(config['training']['learning_rate_bert'])},
        {'params': model.toxicity_head.parameters(), 'lr': float(config['training']['learning_rate_heads'])},
        {'params': model.emotion_head.parameters(), 'lr': float(config['training']['learning_rate_heads'])},
        {'params': uncertainty_loss.parameters(), 'lr': float(config['training']['learning_rate_heads'])}
    ], weight_decay=config['training']['weight_decay'])
    
    # Data
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder'])
    
    # Use only OLID for training loop demonstration (In real AURA we interleave GoEmotions)
    # For simplicity/clarity as requested: Let's train on OLID train, validate on OLID val
    train_dataset = AURADataset("data/processed/olid_train.csv", tokenizer, config['data']['max_length'])
    val_dataset = AURADataset("data/processed/olid_validation.csv", tokenizer, config['data']['max_length'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'])
    
    # Checkpoints
    os.makedirs("outputs/models", exist_ok=True)
    best_f1 = 0.0
    
    # Training Loop
    epochs = config['training']['epochs']
    for epoch in range(1, epochs+1):
        print(f"\nExample Training Epoch {epoch}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, uncertainty_loss, device, epoch)
        print(f"Epoch {epoch} Loss: {train_loss:.4f}")
        
        f1, report = evaluate(model, val_loader, device)
        print(f"Validation Macro-F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"outputs/models/{args.model_name}_best.pt")
            print("🚀 New Best Model Saved!")
            
    print(f"\nTraining Complete. Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
