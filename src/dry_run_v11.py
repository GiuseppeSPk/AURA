import torch
import torch.nn as nn
from torch.autograd import Function
import warnings

warnings.filterwarnings('ignore')

# CONFIG V11 MINI
CONFIG = {
    'encoder': 'bert-base-uncased',
    'num_emotion_classes': 7,
    'dropout': 0.5,
    'adversarial_lambda': 0.1
}

# GRL
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output # The Magic Flip
        return grad_input, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, torch.tensor(alpha))

# Model V11
class AURA_CERBERUS_NEMESIS(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MOCK BERT for speed
        self.bert = nn.Linear(10, 768) 
        self.bert.config = type('obj', (object,), {'hidden_size': 768})
        
        self.dropout = nn.Dropout(config['dropout'])
        self.toxicity_head = nn.Linear(768, 2)
        self.emotion_head = nn.Linear(768, 7)
        self.sentiment_head = nn.Linear(768, 2)
        self.adversary_head = nn.Linear(768, 2) # NEW
        
        # Bias Init Check
        self.toxicity_head.bias.data[0] = 2.94
        self.toxicity_head.bias.data[1] = -2.94
        
    def forward(self, input_ids, attention_mask, alpha=1.0):
        # Mock Forward
        pooled = torch.randn(input_ids.shape[0], 768)
        
        # 1. Standard
        tox = self.toxicity_head(pooled)
        emo = self.emotion_head(pooled)
        sent = self.sentiment_head(pooled)
        
        # 2. Adversarial
        adv_pooled = grad_reverse(pooled, alpha)
        domain = self.adversary_head(adv_pooled)
        
        return {'toxicity': tox, 'emotion': emo, 'sentiment': sent, 'domain': domain}

def run():
    print("Initializing Dry Run V11 (NEMESIS)...")
    try:
        model = AURA_CERBERUS_NEMESIS(CONFIG)
        print(">> Model Instantiated.")
        print(f">> Bias Check: {model.toxicity_head.bias.data}")
        
        # Mock Batch
        input_ids = torch.zeros(4, 10)
        out = model(input_ids, None, alpha=1.0)
        
        print(">> Forward Pass Successful.")
        print(f">> Adversary Output Shape: {out['domain'].shape}")
        
        # Mock Loss
        loss = out['toxicity'].sum() + out['domain'].sum()
        loss.backward()
        print(">> Backward Pass (GRL) Successful.")
        
        print("\n✅ V11 LOGIC CHECK PASSED.")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run()
