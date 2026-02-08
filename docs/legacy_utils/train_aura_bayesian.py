import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

# --- CONFIGURATION ---
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 64,
    'batch_size': 8,
    'epochs': 1,
    'mc_samples': 5,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

# --- MODEL DEFINITION: AURA BAYESIAN (CORRECTED) ---
# Uses HOMOSCEDASTIC (Task-Level) Variance as per Kendall 2018 for MTL weighting.
class AURA_Bayesian(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(CONFIG['model_name'])
        self.dropout = nn.Dropout(0.1)
        
        # Head 1: Toxicity (2 classes)
        self.tox_linear = nn.Linear(768, 2)
        
        # Head 2: Emotion (7 classes)
        self.emo_linear = nn.Linear(768, 7)
        
        # HOMOSCEDASTIC Task-Level Log Variance (Learnable Parameter, NOT per-sample)
        # Initialized to 0 => sigma = exp(0) = 1
        self.tox_log_var = nn.Parameter(torch.zeros(1))
        self.emo_log_var = nn.Parameter(torch.zeros(1))
        
    def forward(self, ids, mask):
        o = self.bert(ids, attention_mask=mask).pooler_output
        o = self.dropout(o)
        
        tox_logits = self.tox_linear(o)
        emo_logits = self.emo_linear(o)
        
        # Return logits and the TASK-LEVEL log_var (expanded for batch compatibility)
        return tox_logits, self.tox_log_var, emo_logits, self.emo_log_var

# --- LOSS FUNCTION: MONTE CARLO INTEGRATION (CORRECTED) ---
# Kendall 2018 (Eq 12) with Regularization Term
def monte_carlo_uncertainty_loss(logits, log_var, targets, T=5):
    """
    Computes the Bayesian uncertainty loss via Monte Carlo sampling.
    Args:
        logits: [B, C] - Raw model predictions.
        log_var: [1] - Task-level log variance (homoscedastic).
        targets: [B] - Ground truth class indices.
        T: int - Number of Monte Carlo samples.
    Returns:
        loss: scalar - Combined MC loss + Regularization.
    """
    batch_size = logits.size(0)
    num_classes = logits.size(1)
    
    # STABILITY FIX: Clamp log_var to prevent overflow/underflow
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped) # sigma = sqrt(exp(log_var))
    
    # 1. Expand logits for T samples: [T, B, C]
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    
    # 2. Add Gaussian Noise (scaled by sigma)
    # Note: For homoscedastic, std is a scalar, broadcast across all samples
    noise = torch.randn_like(logits_expanded).to(logits.device)
    corrupted_logits = logits_expanded + (noise * std)
    
    # 3. Compute Softmax Probabilities for each sample
    probs = F.softmax(corrupted_logits, dim=-1) # [T, B, C]
    
    # 4. Average Probabilities over T samples
    avg_probs = torch.mean(probs, dim=0) # [B, C]
    
    # 5. NLL Loss on Averaged Probabilities
    log_probs = torch.log(avg_probs + 1e-8) # Numerical stability
    mc_loss = F.nll_loss(log_probs, targets)
    
    # 6. REGULARIZATION TERM (Kendall's missing piece!)
    # Penalizes high uncertainty (prevents model from always saying "I don't know")
    # This is 0.5 * log(sigma^2) = 0.5 * log_var
    regularization = 0.5 * log_var_clamped
    
    return mc_loss + regularization

# --- DUMMY DATASET FOR VERIFICATION ---
class MockDataset(Dataset):
    def __init__(self, size=32):
        self.size = size
        self.tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
        self.texts = ["I hate you", "I love you", "Neutral statement", "Ambiguous noise"] * 8
        self.tox_labels = [1, 0, 0, 0] * 8
        self.emo_labels = [0, 3, 6, 6] * 8
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], 
                             max_length=CONFIG['max_length'], 
                             padding='max_length', 
                             truncation=True, 
                             return_tensors='pt')
        return {
            'ids': enc['input_ids'].flatten(),
            'mask': enc['attention_mask'].flatten(),
            'tox': torch.tensor(self.tox_labels[idx]),
            'emo': torch.tensor(self.emo_labels[idx])
        }

# --- TRAINING LOOP VERIFICATION ---
def train_check():
    print(f"🚀 Initializing AURA Bayesian (CORRECTED) on {CONFIG['device']}...")
    model = AURA_Bayesian().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    dataset = MockDataset()
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'])
    
    print("🔄 Starting Training Loop (1 Epoch)...")
    model.train()
    
    for i, batch in enumerate(loader):
        ids = batch['ids'].to(CONFIG['device'])
        mask = batch['mask'].to(CONFIG['device'])
        tox_targets = batch['tox'].to(CONFIG['device'])
        emo_targets = batch['emo'].to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        # Forward
        tox_l, tox_v, emo_l, emo_v = model(ids, mask)
        
        # Loss Calculation (Bayesian with Regularization)
        loss_tox = monte_carlo_uncertainty_loss(tox_l, tox_v, tox_targets, T=CONFIG['mc_samples'])
        loss_emo = monte_carlo_uncertainty_loss(emo_l, emo_v, emo_targets, T=CONFIG['mc_samples'])
        
        total_loss = loss_tox + loss_emo
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        sigma_tox = torch.exp(0.5 * model.tox_log_var).item()
        sigma_emo = torch.exp(0.5 * model.emo_log_var).item()
        print(f"   Step {i}: Loss={total_loss.item():.4f} | Sigma_Tox={sigma_tox:.3f}, Sigma_Emo={sigma_emo:.3f}")

    print("\n✅ VERIFICATION COMPLETE (CORRECTED VERSION)")
    print(f"   Final Sigma_Tox: {torch.exp(0.5 * model.tox_log_var).item():.4f}")
    print(f"   Final Sigma_Emo: {torch.exp(0.5 * model.emo_log_var).item():.4f}")
    print("   If Sigma < 1, the model is CONFIDENT. If Sigma > 1, the model is UNCERTAIN.")

if __name__ == "__main__":
    train_check()
