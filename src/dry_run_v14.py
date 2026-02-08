"""
AURA V14 PHOENIX - Dry Run Verification
Verifica che l'architettura funzioni localmente prima di Kaggle.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Mini CONFIG
CONFIG = {
    'num_emotion_classes': 7,
    'dropout': 0.3,
}

class TaskAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5
        
    def forward(self, hidden_states, attention_mask):
        cls_token = hidden_states[:, 0:1, :]
        Q = self.query(cls_token)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, V)
        return output.squeeze(1), attn_weights.squeeze(1)

class MockRoBERTa(nn.Module):
    """Mock RoBERTa for local testing without downloading."""
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        self.config = type('obj', (object,), {'hidden_size': 768})()
        self.linear = nn.Linear(10, 768)
        
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        # Generate mock hidden states
        hidden = torch.randn(batch_size, seq_len, 768)
        return type('obj', (object,), {'last_hidden_state': hidden})()

class AURA_PHOENIX_Mock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta = MockRoBERTa()
        hidden = 768
        
        self.tox_attention = TaskAttention(hidden)
        self.emo_attention = TaskAttention(hidden)
        self.sent_attention = TaskAttention(hidden)
        self.report_attention = TaskAttention(hidden)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.toxicity_head = nn.Linear(hidden, 2)
        self.emotion_head = nn.Linear(hidden, config['num_emotion_classes'])
        self.sentiment_head = nn.Linear(hidden, 2)
        self.reporting_head = nn.Linear(hidden, 1)
        
        # Bias Init
        with torch.no_grad():
            self.toxicity_head.bias[0] = 2.5
            self.toxicity_head.bias[1] = -2.5
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        hidden_states = outputs.last_hidden_state
        
        tox_repr, tox_attn = self.tox_attention(hidden_states, attention_mask)
        emo_repr, emo_attn = self.emo_attention(hidden_states, attention_mask)
        sent_repr, sent_attn = self.sent_attention(hidden_states, attention_mask)
        report_repr, report_attn = self.report_attention(hidden_states, attention_mask)
        
        tox_repr = self.dropout(tox_repr)
        emo_repr = self.dropout(emo_repr)
        sent_repr = self.dropout(sent_repr)
        report_repr = self.dropout(report_repr)
        
        return {
            'toxicity': self.toxicity_head(tox_repr),
            'emotion': self.emotion_head(emo_repr),
            'sentiment': self.sentiment_head(sent_repr),
            'reporting': self.reporting_head(report_repr).squeeze(-1),
        }

def run():
    print("=" * 50)
    print("AURA V14 PHOENIX - DRY RUN")
    print("=" * 50)
    
    try:
        # 1. Initialize model
        print("\n1. Initializing model...")
        model = AURA_PHOENIX_Mock(CONFIG)
        print("   ✓ Model created successfully")
        
        # 2. Check bias initialization
        print("\n2. Checking bias initialization...")
        bias = model.toxicity_head.bias.data
        print(f"   Toxicity bias: {bias.tolist()}")
        assert bias[0] > bias[1], "Bias should favor non-toxic!"
        print("   ✓ Bias initialization correct")
        
        # 3. Forward pass
        print("\n3. Testing forward pass...")
        batch_size, seq_len = 4, 32
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        out = model(input_ids, attention_mask)
        print(f"   Toxicity shape: {out['toxicity'].shape}")
        print(f"   Emotion shape: {out['emotion'].shape}")
        print(f"   Sentiment shape: {out['sentiment'].shape}")
        print(f"   Reporting shape: {out['reporting'].shape}")
        print("   ✓ Forward pass successful")
        
        # 4. Loss computation
        print("\n4. Testing loss computation...")
        tox_loss = F.cross_entropy(out['toxicity'], torch.zeros(batch_size, dtype=torch.long))
        emo_loss = F.binary_cross_entropy_with_logits(out['emotion'], torch.zeros(batch_size, 7))
        sent_loss = F.cross_entropy(out['sentiment'], torch.ones(batch_size, dtype=torch.long))
        report_loss = F.binary_cross_entropy_with_logits(out['reporting'], torch.zeros(batch_size))
        total_loss = tox_loss + emo_loss + sent_loss + report_loss
        print(f"   Total loss: {total_loss.item():.4f}")
        print("   ✓ Loss computation successful")
        
        # 5. Backward pass
        print("\n5. Testing backward pass...")
        total_loss.backward()
        print("   ✓ Backward pass successful")
        
        # 6. Parameter count
        print("\n6. Model statistics...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable: {trainable:,}")
        
        print("\n" + "=" * 50)
        print("✅ ALL CHECKS PASSED - V14 READY FOR KAGGLE")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run()
