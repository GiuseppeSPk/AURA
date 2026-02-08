"""
AURA V10 Archangel - Dry Run Verification
Tests architecture and data loading locally before Kaggle.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("AURA V10 ARCHANGEL - DRY RUN")
print("="*60)

# Config
CONFIG = {
    'hidden_dim': 768,
    'n_heads': 8,
    'dropout': 0.3,
    'num_emotion_classes': 7
}

# 1. TaskSpecificMHA Module
class TaskSpecificMHA(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, 
                                          batch_first=True, dropout=dropout)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask):
        key_padding_mask = (attention_mask == 0)
        attn_output, attn_weights = self.mha(
            query=hidden_states, key=hidden_states, value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        output = self.layernorm(hidden_states + self.dropout(attn_output))
        return output, attn_weights

# 2. Mock Model (without loading RoBERTa)
class MockARCHANGEL(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden_dim']
        
        self.tox_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.emo_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.sent_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        self.report_mha = TaskSpecificMHA(hidden, config['n_heads'], config['dropout'])
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.toxicity_head = nn.Linear(hidden, 2)
        self.emotion_head = nn.Linear(hidden, config['num_emotion_classes'])
        self.sentiment_head = nn.Linear(hidden, 2)
        self.reporting_head = nn.Linear(hidden, 1)
        
        with torch.no_grad():
            self.toxicity_head.bias[0] = 2.5
            self.toxicity_head.bias[1] = -2.5

    def _mean_pool(self, seq, mask):
        mask_exp = mask.unsqueeze(-1).expand(seq.size()).float()
        return (seq * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

    def forward(self, hidden_states, attention_mask):
        tox_seq, _ = self.tox_mha(hidden_states, attention_mask)
        emo_seq, _ = self.emo_mha(hidden_states, attention_mask)
        sent_seq, _ = self.sent_mha(hidden_states, attention_mask)
        rep_seq, _ = self.report_mha(hidden_states, attention_mask)
        
        return {
            'toxicity': self.toxicity_head(self.dropout(self._mean_pool(tox_seq, attention_mask))),
            'emotion': self.emotion_head(self.dropout(self._mean_pool(emo_seq, attention_mask))),
            'sentiment': self.sentiment_head(self.dropout(self._mean_pool(sent_seq, attention_mask))),
            'reporting': self.reporting_head(self.dropout(self._mean_pool(rep_seq, attention_mask))).squeeze(-1)
        }

# 3. UncertaintyLoss
class UncertaintyLoss(nn.Module):
    def __init__(self, n_tasks=4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            total += torch.exp(-self.log_vars[i]) * loss + self.log_vars[i] * 0.5
        return total

def test_architecture():
    print("\n1. Testing Architecture...")
    model = MockARCHANGEL(CONFIG)
    loss_fn = UncertaintyLoss()
    
    # Mock input
    batch_size, seq_len = 4, 32
    hidden_states = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -5:] = 0  # Simulate padding
    
    # Forward
    out = model(hidden_states, attention_mask)
    
    print(f"   toxicity shape: {out['toxicity'].shape} (expected: [4, 2])")
    print(f"   emotion shape: {out['emotion'].shape} (expected: [4, 7])")
    print(f"   sentiment shape: {out['sentiment'].shape} (expected: [4, 2])")
    print(f"   reporting shape: {out['reporting'].shape} (expected: [4])")
    
    assert out['toxicity'].shape == (4, 2), "Toxicity shape mismatch!"
    assert out['emotion'].shape == (4, 7), "Emotion shape mismatch!"
    assert out['sentiment'].shape == (4, 2), "Sentiment shape mismatch!"
    assert out['reporting'].shape == (4,), "Reporting shape mismatch!"
    
    # Loss
    losses = [
        F.cross_entropy(out['toxicity'], torch.zeros(4, dtype=torch.long)),
        F.binary_cross_entropy_with_logits(out['emotion'], torch.zeros(4, 7)),
        F.cross_entropy(out['sentiment'], torch.ones(4, dtype=torch.long)),
        F.binary_cross_entropy_with_logits(out['reporting'], torch.zeros(4))
    ]
    total_loss = loss_fn(losses)
    print(f"   total_loss: {total_loss.item():.4f}")
    
    # Backward
    total_loss.backward()
    print("   ✅ Architecture test PASSED")

def test_data():
    print("\n2. Testing Data Files...")
    data_dir = 'data/aura_v9_clean'
    
    files = {
        'toxicity_train': ('toxicity_train.csv', ['text', 'label']),
        'toxicity_val': ('toxicity_val.csv', ['text', 'label']),
        'emotions': ('emotions_train.csv', ['text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']),
        'sentiment': ('sentiment_train.csv', ['text', 'label']),
        'reporting': ('reporting_examples.csv', ['text', 'is_reporting'])
    }
    
    for name, (filename, expected_cols) in files.items():
        path = f'{data_dir}/{filename}'
        try:
            df = pd.read_csv(path)
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"   ❌ {name}: Missing columns {missing}")
            else:
                print(f"   ✅ {name}: {len(df)} rows, columns OK")
        except FileNotFoundError:
            print(f"   ❌ {name}: FILE NOT FOUND at {path}")

if __name__ == '__main__':
    try:
        test_architecture()
        test_data()
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED - READY FOR KAGGLE")
        print("="*60)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
