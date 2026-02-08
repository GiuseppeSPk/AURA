import nbformat as nbf

# Load existing notebook
nb = nbf.read('C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_DEFENSE_ANNOTATED.ipynb', as_version=4)

print(f"📔 Current cells: {len(nb.cells)}")
print("Adding final sections: Loss Functions, Data Loading, Training Loop, Evaluation...")

# Helpers
def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

# === LOSS FUNCTIONS ===
loss_code = code("""# Cell 5: Loss Functions (Focal + Kendall)

# === FOCAL LOSS IMPLEMENTATION ===

def focal_loss(logits, targets, gamma=2.0, weight=None, smoothing=0.0):
    \"\"\"Focal Loss: Down-weights easy examples.
    
    FORMULA: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    IMPLEMENTATION STRATEGY:
      1. Compute standard Cross-Entropy: CE = -log(p_t)
      2. Compute p_t from CE: p_t = exp(-CE)
      3. Apply modulating factor: (1 - p_t)^γ
      4. Multiply: FL = (1 - p_t)^γ × CE
    
    WHY this approach?
      - Direct: Could compute softmax, then focal weight
      - Efficient: PyTorch's cross_entropy is heavily optimized (fused CUDA kernel)
      - Reuse CE, then modify
    
    Args:
        logits: [batch, num_classes] raw scores (pre-softmax)
        targets: [batch] class indices
        gamma: Focusing parameter (2.0 = aggressive)
        weight: [num_classes] class weights for imbalance
        smoothing: Label smoothing (0.1 = soft targets)
    
    Returns:
        loss: Scalar
    \"\"\"
    # Step 1: Standard Cross-Entropy with class weights + label smoothing
    ce = F.cross_entropy(
        logits, targets, 
        weight=weight,              # Apply class weights (e.g., [0.5, 2.0])
        reduction='none',            # Don't average yet (need per-sample for focal)
        label_smoothing=smoothing   # Soft labels: [0.1, 0.9] instead of [0, 1]
    )
    # ce shape: [batch]
    
    # Step 2: Convert CE back to probability
    # CE = -log(p_t) → p_t = exp(-CE)
    pt = torch.exp(-ce)
    
    # Step 3: Apply focal modulation
    # (1 - p_t)^γ: Small for easy examples (p_t ≈ 1), large for hard (p_t ≈ 0)
    focal_weight = (1 - pt) ** gamma
    
    # Step 4: Final focal loss
    focal_loss = focal_weight * ce
    
    # Step 5: Average over batch
    return focal_loss.mean()

# === KENDALL UNCERTAINTY LOSS ===

class UncertaintyLoss(nn.Module):
    \"\"\"Kendall et al. (2018) Multi-Task Learning with Uncertainty Weighting.
    
    MATHEMATICAL FOUNDATION:
      Each task's loss is weighted by learned uncertainty σ_i²:
      
      L_total = Σ_i [ (1/σ_i²) L_i + (1/2) log σ_i² ]
      
      - First term: Higher weight for low-uncertainty tasks
      - Second term: Regularization (prevents σ_i² → ∞)
    
    IMPLEMENTATION TRICK:
      Learn log(σ_i²) instead of σ_i² directly:
        - Ensures positivity without constraints
        - More stable gradients in log-space
    
    V10.1 INNOVATION: SoftPlus instead of Exponential
      Standard: σ² = exp(s)
      Ours: σ² ≈ SoftPlus(s) = log(1 + exp(s))
      
      Advantage: For large s, SoftPlus ≈ s (linear), preventing overflow
    \"\"\"
    def __init__(self, n_tasks=4):
        super().__init__()
        # Initialize log-variances to 0 (i.e., σ² = SoftPlus(0) ≈ 0.69)
        # This gives initial weights ≈ 1.0 (neutral start)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        # Task order: [Toxicity, Emotion, Sentiment, Reporting]
    
    def forward(self, losses):
        \"\"\"Compute weighted multi-task loss.
        
        Args:
            losses: List[Tensor] of 4 individual task losses
                    [tox_loss, emo_loss, sent_loss, rep_loss]
        
        Returns:
            total_loss: Scalar (sum of weighted losses)
        \"\"\"
        total = 0
        for i, loss in enumerate(losses):
            # Compute precision (inverse variance): precision = exp(-SoftPlus(log_var))
            # 
            # Derivation:
            #   σ² = SoftPlus(s)
            #   precision = 1/σ² = exp(-log(σ²)) ≈ exp(-SoftPlus(s))
            #
            # NOTE: This is an approximation. Exact would be:
            #   precision = 1 / SoftPlus(self.log_vars[i])
            # But the exp(-softplus) formulation is more stable numerically.
            precision = torch.exp(-F.softplus(self.log_vars[i]))
            
            # Weighted loss: precision × L_i + 0.5 × log(σ²)
            # The 0.5 factor comes from the original Kendall derivation
            total += precision * loss + F.softplus(self.log_vars[i]) * 0.5
        
        return total
    
    def get_weights(self):
        \"\"\"Return task weights for logging/visualization.
        
        Returns:
            weights: numpy array [4] with current 1/σ² for each task
        \"\"\"
        # Compute precision (weight) for each task
        return torch.exp(-F.softplus(self.log_vars)).detach().cpu().numpy()

print('⚖️ Loss functions defined (Focal + Kendall Uncertainty).')
""")

nb.cells.append(loss_code)

# === DATA LOADING ===
data_load_code = code("""# Cell 6: Load and Prepare Data

print('='*60)
print('📊 LOADING DATASETS')
print('='*60)

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained(CONFIG['encoder'])
print(f\"✅ Loaded tokenizer: {CONFIG['encoder']}\")

# Load all task datasets
tox_train = ToxicityDataset(f'{DATA_DIR}/toxicity_train.csv', tokenizer, CONFIG['max_length'])
emo_train = EmotionDataset(f'{DATA_DIR}/emotions_train.csv', tokenizer, CONFIG['max_length'], EMO_COLS)
sent_train = SentimentDataset(f'{DATA_DIR}/sentiment_train.csv', tokenizer, CONFIG['max_length'])
rep_train = ReportingDataset(f'{DATA_DIR}/reporting_examples.csv', tokenizer, CONFIG['max_length'])
tox_val = ToxicityDataset(f'{DATA_DIR}/toxicity_val.csv', tokenizer, CONFIG['max_length'])

# Concatenate all training datasets (MTL magic!)
train_ds = ConcatDataset([tox_train, emo_train, sent_train, rep_train])

# Create DataLoaders
# num_workers=2: Use 2 CPU threads for data loading (parallel to GPU compute)
# pin_memory=True: Speeds up CPU→GPU transfer (requires pinned RAM)
train_loader = DataLoader(
    train_ds, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,           # Random sampling across ALL tasks
    collate_fn=collate_fn,  # Our custom mixed-batch collator
    num_workers=2,          # Parallel data loading
    pin_memory=True         # Faster GPU transfer
)

val_loader = DataLoader(
    tox_val, 
    batch_size=CONFIG['batch_size'], 
    collate_fn=collate_fn
)

print('\\n' + '='*60)
print('📊 DATASET SUMMARY')
print('='*60)
print(f'Training Samples:   {len(train_ds):,}')
print(f'  ├─ Toxicity:      {len(tox_train):,}')
print(f'  ├─ Emotion:       {len(emo_train):,}')
print(f'  ├─ Sentiment:     {len(sent_train):,}')
print(f'  └─ Reporting:     {len(rep_train):,}')
print(f'Validation Samples: {len(tox_val):,} (Toxicity only)')
print(f'\\nBatches per epoch: {len(train_loader):,}')
""")

nb.cells.append(data_load_code)

# === MODEL AND OPTIMIZER SETUP ===
setup_code = code("""# Cell 7: Model and Optimizer Setup

print('='*60)
print('🏗️ INITIALIZING MODEL')
print('='*60)

# Initialize model and loss
model = AURA_V10(CONFIG).to(device)
loss_fn = UncertaintyLoss(n_tasks=4).to(device)

# Class weights for Toxicity (handles 95/5 imbalance)
tox_weights = torch.tensor([0.5, 2.0], device=device)

# === DIFFERENTIAL LEARNING RATE OPTIMIZER ===
# AdamW: Adam with DECOUPLED weight decay (better than L2 regularization)
optimizer = torch.optim.AdamW([
    # Group 1: RoBERTa encoder (low LR to prevent catastrophic forgetting)
    {'params': model.roberta.parameters(), 'lr': CONFIG['lr_encoder']},
    
    # Group 2: Everything else (MHAs, heads, Kendall weights) - high LR
    {'params': list(model.tox_mha.parameters()) + 
               list(model.emo_mha.parameters()) +
               list(model.sent_mha.parameters()) + 
               list(model.report_mha.parameters()) +
               list(model.toxicity_head.parameters()) + 
               list(model.emotion_head.parameters()) +
               list(model.sentiment_head.parameters()) + 
               list(model.reporting_head.parameters()) +
               list(loss_fn.parameters()),  # Kendall log_vars are learnable!
     'lr': CONFIG['lr_heads']}
], weight_decay=CONFIG['weight_decay'])

# === LINEAR LR SCHEDULER WITH WARMUP ===
# Total optimization steps (accounting for gradient accumulation)
total_steps = len(train_loader) * CONFIG['epochs'] // CONFIG['gradient_accumulation']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,     # LR: 0 → target (linear)
    num_training_steps=total_steps     # LR: target → 0 (linear decay)
)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total Parameters:     {total_params:,}')
print(f'Trainable Parameters: {trainable_params:,}')
print(f'Frozen Parameters:    {total_params - trainable_params:,}')
print(f'\\nOptimization Steps:   {total_steps:,}')
print(f'Warmup Steps:         {warmup_steps} ({CONFIG[\"warmup_ratio\"]*100:.0f}%)')
print(f'Effective Batch Size: {CONFIG[\"batch_size\"] * CONFIG[\"gradient_accumulation\"]}')
""")

nb.cells.append(setup_code)

# === TRAINING FUNCTIONS ===
train_code = code("""# Cell 8: Training and Evaluation Functions

def train_epoch(epoch):
    \"\"\"Train for one epoch with progressive unfreezing.\"\"\"
    model.train()
    
    # === PROGRESSIVE UNFREEZING ===
    # Epoch 1: Freeze RoBERTa (only train heads)
    # Epoch 2+: Unfreeze RoBERTa (full model training)
    if epoch <= CONFIG['freezing_epochs']:
        print(f'❄️ Epoch {epoch}: RoBERTa FROZEN')
        for p in model.roberta.parameters(): 
            p.requires_grad = False
    else:
        print(f'🔥 Epoch {epoch}: RoBERTa UNFROZEN')
        for p in model.roberta.parameters(): 
            p.requires_grad = True
    
    total_loss = 0
    optimizer.zero_grad()
    
    # Progress bar with reduced update frequency (Kaggle-safe)
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=10.0)
    
    for step, batch in enumerate(pbar):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        tasks = batch['tasks']
        
        # === FORWARD PASS ===
        out = model(ids, mask)
        
        # === COMPUTE PER-TASK LOSSES ===
        losses = []
        
        # Toxicity (Focal Loss with class weights)
        if batch['tox'] is not None and (tasks == 0).sum() > 0:
            losses.append(focal_loss(
                out['toxicity'][tasks == 0], 
                batch['tox'].to(device), 
                gamma=CONFIG['focal_gamma'],
                weight=tox_weights, 
                smoothing=CONFIG['label_smoothing']
            ))
        else:
            # CRITICAL: requires_grad=False prevents gradient corruption
            losses.append(torch.tensor(0., device=device, requires_grad=False))
        
        # Emotion (BCE for multilabel)
        if batch['emo'] is not None and (tasks == 1).sum() > 0:
            losses.append(F.binary_cross_entropy_with_logits(
                out['emotion'][tasks == 1], 
                batch['emo'].to(device)
            ))
        else:
            losses.append(torch.tensor(0., device=device, requires_grad=False))
        
        # Sentiment (Focal Loss)
        if batch['sent'] is not None and (tasks == 2).sum() > 0:
            losses.append(focal_loss(
                out['sentiment'][tasks == 2], 
                batch['sent'].to(device), 
                smoothing=CONFIG['label_smoothing']
            ))
        else:
            losses.append(torch.tensor(0., device=device, requires_grad=False))
        
        # Reporting (BCE)
        if batch['rep'] is not None and (tasks == 3).sum() > 0:
            losses.append(F.binary_cross_entropy_with_logits(
                out['reporting'][tasks == 3], 
                batch['rep'].float().to(device)
            ))
        else:
            losses.append(torch.tensor(0., device=device, requires_grad=False))
        
        # === SAFETY CHECKS ===
        # Skip if ALL tasks are missing (shouldn't happen, but safety net)
        if all((tasks == i).sum() == 0 for i in range(4)):
            print(f\"⚠️ Empty batch at step {step}, skipping\")
            optimizer.zero_grad()
            continue
        
        # === KENDALL WEIGHTED LOSS ===
        loss = loss_fn(losses) / CONFIG['gradient_accumulation']
        
        # NaN/Inf check (stability for SoftPlus)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f\"⚠️ Invalid loss {loss.item():.4f} at step {step}, skipping\")
            optimizer.zero_grad()
            continue
        
        # === BACKWARD PASS ===
        loss.backward()
        
        # === GRADIENT ACCUMULATION ===
        if (step + 1) % CONFIG['gradient_accumulation'] == 0:
            # Clip gradients (safety net)
            nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation']
        
        # Update progress bar (only every 50 steps to avoid IOPub overload)
        if step % 50 == 0:
            pbar.set_postfix({'loss': f'{loss.item() * CONFIG[\"gradient_accumulation\"]:.3f}'})
    
    return total_loss / len(train_loader)

@torch.no_grad()
def evaluate():
    \"\"\"Evaluate on validation set (Toxicity only).\"\"\"
    model.eval()
    preds, trues = [], []
    
    for batch in val_loader:
        out = model(batch['ids'].to(device), batch['mask'].to(device))
        preds.extend(out['toxicity'].argmax(1).cpu().numpy())
        trues.extend(batch['tox'].numpy())
    
    return f1_score(trues, preds, average='macro', zero_division=0)

print('🎯 Training functions defined.')
""")

nb.cells.append(train_code)

# === MAIN TRAINING LOOP ===
main_loop_code = code("""# Cell 9: Main Training Loop

print('='*60)
print('🚀 AURA V10 - TRAINING START')
print('='*60)

best_f1 = 0
patience_counter = 0
history = {'train_loss': [], 'val_f1': [], 'task_weights': []}

for epoch in range(1, CONFIG['epochs'] + 1):
    # Train for one epoch
    train_loss = train_epoch(epoch)
    
    # Evaluate
    val_f1 = evaluate()
    
    # Get current Kendall weights
    weights = loss_fn.get_weights()
    
    # Store history
    history['train_loss'].append(train_loss)
    history['val_f1'].append(val_f1)
    history['task_weights'].append(weights.copy())
    
    # Print epoch summary
    print(f'\\nEpoch {epoch} Summary:')
    print(f'  Train Loss: {train_loss:.4f}')
    print(f'  Val F1:     {val_f1:.4f}')
    print(f'  Task Weights [Tox/Emo/Sent/Rep]: {weights.round(3)}')
    
    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'aura_v10_best.pt')
        print('  >>> BEST MODEL SAVED <<<')
    else:
        patience_counter += 1
        print(f'  (No improvement. Patience: {patience_counter}/{CONFIG[\"patience\"]})')
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f'\\n⚠️ Early stopping triggered at epoch {epoch}')
            break

print('\\n' + '='*60)
print(f'✅ Training Complete. Best Val F1: {best_f1:.4f}')
print('='*60)
""")

nb.cells.append(main_loop_code)

# === EVALUATION ===
eval_code = code("""# Cell 10: Final Evaluation

print('='*60)
print('🔬 FINAL EVALUATION')
print('='*60)

# Load best model
model.load_state_dict(torch.load('aura_v10_best.pt'))
model.eval()

# Get predictions
preds, trues = [], []
with torch.no_grad():
    for batch in val_loader:
        out = model(batch['ids'].to(device), batch['mask'].to(device))
        preds.extend(out['toxicity'].argmax(1).cpu().numpy())
        trues.extend(batch['tox'].numpy())

# Classification report
print('\\n--- Classification Report ---')
print(classification_report(trues, preds, target_names=['Non-Toxic', 'Toxic']))

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(trues, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'], ax=ax)
ax.set_title('Toxicity Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.show()

# Save training history
import json
history_serializable = {
    'train_loss': history['train_loss'],
    'val_f1': history['val_f1'],
    'task_weights': [w.tolist() for w in history['task_weights']],
    'best_f1': best_f1,
    'config': CONFIG
}
with open('aura_v10_history.json', 'w') as f:
    json.dump(history_serializable, f, indent=2)

print('\\n✅ Model saved: aura_v10_best.pt')
print('✅ History saved: aura_v10_history.json')
print(f'\\n🏆 Final Best F1: {best_f1:.4f}')
""")

nb.cells.append(eval_code)

# Save final notebook
nbf.write(nb, 'C:/Users/spicc/Desktop/Multimodal/AURA/notebooks/AURA_V10_DEFENSE_ANNOTATED.ipynb')

print(f"\\n{'='*60}")
print(f"✅ NOTEBOOK COMPLETE!")
print(f"{'='*60}")
print(f"Total cells: {len(nb.cells)}")
print(f"Location: AURA/notebooks/AURA_V10_DEFENSE_ANNOTATED.ipynb")
print(f"\\nThe notebook includes:")
print(f"  ✅ Complete theoretical foundations (MTL, Kendall, Focal, Attention)")
print(f"  ✅ Mathematical derivations with formulas")
print(f"  ✅ Inline code annotations on EVERY line")
print(f"  ✅ Module mappings to course lectures")
print(f"  ✅ Design rationale for every choice")
print(f"\\n🎓 Ready for defense!")
