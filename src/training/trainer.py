"""
AURA Trainer - Production Ready for Kaggle T4
==============================================
Features:
- Focal Loss + Class Weights for imbalanced data
- Uncertainty Weighting for MTL
- Early Stopping with patience
- LR Warmup + Cosine Decay
- Gradient Accumulation for larger effective batch size
- Mixed Precision (AMP) for T4 Tensorcores
- Robust error handling and checkpointing
"""

import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from pathlib import Path

# Handle different PyTorch versions for AMP
try:
    from torch.cuda.amp import GradScaler, autocast
except ImportError:
    from torch.amp import GradScaler, autocast

from src.models.aura import AURA
from src.data.aura_dataset import AURADataset
from src.models.losses import UncertaintyLoss, FocalLoss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def load_class_weights(path, device):
    """Load class weights from JSON file."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        weights = torch.tensor(data['weights'], dtype=torch.float32, device=device)
        print(f"✅ Loaded class weights: {weights.tolist()}")
        return weights
    else:
        print(f"⚠️ Class weights not found at {path}, using uniform weights")
        return None


def train_one_epoch(model, loader, optimizer, scheduler, uncertainty_loss, 
                    focal_loss_tox, focal_loss_emo, device, epoch, scaler,
                    gradient_accumulation_steps=2):
    """
    Training loop with:
    - Focal Loss for class imbalance
    - Gradient Accumulation for larger effective batch
    - Mixed Precision for T4/P100
    """
    model.train()
    total_loss = 0
    total_loss_tox = 0
    total_loss_emo = 0
    num_batches = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loop):
        # Move batch to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        tox_targets = batch['toxicity_target'].to(device, non_blocking=True)
        emo_targets = batch['emotion_target'].to(device, non_blocking=True)
        
        # Mixed Precision Forward Pass
        with autocast(device_type='cuda', dtype=torch.float16):
            tox_logits, emo_logits = model(input_ids, attention_mask)
            
            # Toxicity Loss with Focal Loss
            tox_mask = tox_targets != -1
            if tox_mask.sum() > 0:
                loss_t = focal_loss_tox(tox_logits[tox_mask], tox_targets[tox_mask])
            else:
                loss_t = torch.tensor(0.0, device=device, requires_grad=True)

            # Emotion Loss (BCE for multi-label)
            emo_mask = emo_targets[:, 0] != -1
            if emo_mask.sum() > 0:
                loss_e = nn.BCEWithLogitsLoss()(emo_logits[emo_mask], emo_targets[emo_mask])
            else:
                loss_e = torch.tensor(0.0, device=device, requires_grad=True)
                
            # Combined with Uncertainty Weighting
            loss = uncertainty_loss(loss_t, loss_e)
            
            # Scale for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward with Scaler
        scaler.scale(loss).backward()
        
        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
                
            optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item() * gradient_accumulation_steps
        total_loss_tox += loss_t.item()
        total_loss_emo += loss_e.item()
        num_batches += 1
        
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        loop.set_postfix(
            loss=f"{loss.item()*gradient_accumulation_steps:.4f}",
            loss_t=f"{loss_t.item():.4f}",
            loss_e=f"{loss_e.item():.4f}",
            lr=f"{current_lr:.2e}"
        )
        
    return {
        'total': total_loss / num_batches,
        'toxicity': total_loss_tox / num_batches,
        'emotion': total_loss_emo / num_batches
    }


def evaluate(model, loader, device):
    """Evaluation with detailed metrics."""
    model.eval()
    all_tox_preds = []
    all_tox_true = []
    all_tox_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            tox_targets = batch['toxicity_target'].to(device, non_blocking=True)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                tox_logits, _ = model(input_ids, attention_mask)
            
            probs = torch.softmax(tox_logits.float(), dim=1)
            preds = torch.argmax(probs, dim=1)
            
            mask = tox_targets != -1
            if mask.sum() > 0:
                all_tox_preds.extend(preds[mask].cpu().numpy())
                all_tox_true.extend(tox_targets[mask].cpu().numpy())
                all_tox_probs.extend(probs[mask, 1].cpu().numpy())
    
    if len(all_tox_true) == 0:
        return 0.0, {}
                
    f1 = f1_score(all_tox_true, all_tox_preds, average='macro')
    report = classification_report(all_tox_true, all_tox_preds, output_dict=True)
    cm = confusion_matrix(all_tox_true, all_tox_preds)
    
    return f1, {
        'report': report,
        'confusion_matrix': cm.tolist(),
        'predictions': all_tox_preds,
        'probabilities': all_tox_probs
    }


def main():
    parser = argparse.ArgumentParser(description="AURA Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_name", type=str, default="aura")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("🌟 AURA Training - Production Ready")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_name}")
        
    # Device setup with T4/P100 optimizations
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        
        # Optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Using CPU (very slow).")
    
    # Model
    model = AURA(args.config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Load checkpoint if resuming
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume))
        print(f"✅ Resumed from {args.resume}")
    
    # Loss functions
    uncertainty_loss = UncertaintyLoss().to(device)
    
    # Load class weights for Focal Loss
    tox_weights = load_class_weights("data/processed/olid_class_weights.json", device)
    emo_weights = load_class_weights("data/processed/class_weights.json", device)
    
    # Focal Loss with class weights
    focal_loss_tox = FocalLoss(
        alpha=tox_weights[1] if tox_weights is not None else 1.0,
        gamma=config['loss'].get('focal_gamma', 2.0)
    ).to(device)
    
    focal_loss_emo = FocalLoss(
        gamma=config['loss'].get('focal_gamma', 2.0)
    ).to(device)
    
    # Optimizer with differential LR
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': float(config['training']['learning_rate_bert'])},
        {'params': model.toxicity_head.parameters(), 'lr': float(config['training']['learning_rate_heads'])},
        {'params': model.emotion_head.parameters(), 'lr': float(config['training']['learning_rate_heads'])},
        {'params': uncertainty_loss.parameters(), 'lr': float(config['training']['learning_rate_heads'])}
    ], weight_decay=config['training']['weight_decay'])
    
    # Gradient Scaler for AMP
    scaler = GradScaler()
    
    # Data
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder'])
    
    # Kaggle-safe batch size
    batch_size = min(config['data'].get('batch_size', 16), 16)  # Cap at 16 for T4
    gradient_accumulation = config['training'].get('gradient_accumulation_steps', 2)
    effective_batch_size = batch_size * gradient_accumulation
    print(f"📦 Batch: {batch_size} × {gradient_accumulation} = {effective_batch_size} effective")
    
    train_dataset = AURADataset("data/processed/olid_train.csv", tokenizer, config['data']['max_length'])
    val_dataset = AURADataset("data/processed/olid_validation.csv", tokenizer, config['data']['max_length'])
    
    # DataLoaders with Kaggle-safe settings
    num_workers = 2  # Safe for Kaggle
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=True  # Avoid inconsistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    print(f"📊 Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"📊 Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # LR Scheduler with warmup
    epochs = config['training']['epochs']
    total_steps = len(train_loader) * epochs // gradient_accumulation
    warmup_steps = int(total_steps * config['training'].get('warmup_steps', 0.1))
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[
            float(config['training']['learning_rate_bert']),
            float(config['training']['learning_rate_heads']),
            float(config['training']['learning_rate_heads']),
            float(config['training']['learning_rate_heads'])
        ],
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    print(f"📈 LR Schedule: {warmup_steps} warmup steps, {total_steps} total steps")
    
    # Early stopping
    patience = config['training'].get('patience', 3)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    print(f"⏱️ Early Stopping: patience={patience}")
    
    # Training directory
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    
    best_f1 = 0.0
    training_log = []
    
    print("\n" + "="*60)
    print("🚀 Starting Training")
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, scheduler, uncertainty_loss,
            focal_loss_tox, focal_loss_emo, device, epoch, scaler,
            gradient_accumulation
        )
        
        # Evaluate
        val_f1, val_results = evaluate(model, val_loader, device)
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_losses['total'],
            'train_loss_tox': train_losses['toxicity'],
            'train_loss_emo': train_losses['emotion'],
            'val_f1': val_f1,
            'val_report': val_results.get('report', {})
        }
        training_log.append(log_entry)
        
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_losses['total']:.4f} (tox: {train_losses['toxicity']:.4f}, emo: {train_losses['emotion']:.4f})")
        print(f"   Val Macro-F1: {val_f1:.4f}")
        
        # Check improvement
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_f1': best_f1,
                'config': config
            }, f"outputs/models/{args.model_name}_best.pt")
            print(f"   💾 New best model saved! (F1: {best_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_f1):
            print(f"\n⏹️ Early stopping triggered at epoch {epoch}")
            break
        else:
            print(f"   ⏳ Patience: {early_stopping.counter}/{patience}")
    
    # Save training log
    with open(f"outputs/logs/{args.model_name}_training_log.json", 'w') as f:
        json.dump(training_log, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best Validation F1: {best_f1:.4f}")
    print(f"Model saved to: outputs/models/{args.model_name}_best.pt")
    print(f"Training log: outputs/logs/{args.model_name}_training_log.json")


if __name__ == "__main__":
    main()
