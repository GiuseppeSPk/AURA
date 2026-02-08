"""
AURA V2 DistilBERT - Local Verification Script
Tests all components before Kaggle upload.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertTokenizer

print("="*60)
print("🧪 AURA V2 DISTILBERT VERIFICATION")
print("="*60)

CONFIG = {
    'encoder': 'distilbert-base-uncased',
    'max_length': 64,
    'num_emotion_classes': 7,
    'dropout': 0.3,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

print(f"\n📍 Device: {CONFIG['device']}")

# --- MODEL ---
class AURA_Bayesian_V2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(config['encoder'])
        hidden_size = self.distilbert.config.hidden_size
        self.dropout = nn.Dropout(config['dropout'])
        self.toxicity_head = nn.Linear(hidden_size, 2)
        self.emotion_head = nn.Linear(hidden_size, config['num_emotion_classes'])
        self.tox_log_var = nn.Parameter(torch.zeros(1))
        self.emo_log_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(cls_output)
        tox_logits = self.toxicity_head(pooled)
        emo_logits = self.emotion_head(pooled)
        return tox_logits, emo_logits, self.tox_log_var, self.emo_log_var

print("\n✅ [1/5] V2 Model class defined (DistilBERT)")

# --- LOSS FUNCTION ---
def monte_carlo_uncertainty_loss_classification(logits, log_var, targets, T=5, label_smoothing=0.1):
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped)
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_expanded)
    corrupted_logits = logits_expanded + (noise * std)
    probs = F.softmax(corrupted_logits, dim=-1)
    avg_probs = torch.mean(probs, dim=0)
    log_probs = torch.log(avg_probs + 1e-8)
    
    # Label Smoothing
    n_classes = logits.size(-1)
    one_hot = F.one_hot(targets, n_classes).float()
    smooth_targets = one_hot * (1 - label_smoothing) + label_smoothing / n_classes
    nll = -(smooth_targets * log_probs).sum(dim=-1).mean()
    
    regularization = 0.5 * log_var_clamped
    return nll + regularization

print("✅ [2/5] Loss function with Label Smoothing defined")

# --- INSTANTIATION ---
print("\n🔄 Loading DistilBERT (this may take a moment)...")
model = AURA_Bayesian_V2(CONFIG).to(CONFIG['device'])

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ [3/5] Model loaded: {total_params:,} parameters (should be ~67M)")

# --- FORWARD PASS ---
tokenizer = DistilBertTokenizer.from_pretrained(CONFIG['encoder'])
texts = ["I hate you", "I love you", "Neutral text", "Test sample"]
enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')

input_ids = enc['input_ids'].to(CONFIG['device'])
attention_mask = enc['attention_mask'].to(CONFIG['device'])
targets = torch.tensor([1, 0, 0, 0], device=CONFIG['device'])

tox_logits, emo_logits, tox_log_var, emo_log_var = model(input_ids, attention_mask)

print(f"\n📊 Forward Pass:")
print(f"   Tox Logits shape: {tox_logits.shape}")  # [4, 2]
print(f"   Emo Logits shape: {emo_logits.shape}")  # [4, 7]
print(f"   Dropout: {model.dropout.p}")  # Should be 0.3

print("✅ [4/5] Forward pass successful")

# --- LOSS COMPUTATION ---
loss = monte_carlo_uncertainty_loss_classification(tox_logits, tox_log_var, targets, T=5, label_smoothing=0.1)
print(f"\n📊 Loss (with Label Smoothing 0.1): {loss.item():.4f}")

# --- BACKWARD ---
loss.backward()
print(f"✅ [5/5] Backward pass successful")
print(f"   Gradient on tox_log_var: {model.tox_log_var.grad.item():.6f}")

print("\n" + "="*60)
print("🎉 ALL V2 TESTS PASSED - READY FOR KAGGLE")
print("="*60)
