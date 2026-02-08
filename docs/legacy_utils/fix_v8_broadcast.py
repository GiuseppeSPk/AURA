import json

with open('notebooks/AURA_V8_Colab.ipynb', 'r') as f:
    nb = json.load(f)

# Fix the loss functions cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'def focal_loss_with_uncertainty' in source:
            new_source = '''# 5. Loss Functions
def focal_loss_with_uncertainty(logits, log_var, targets, gamma=2.0, T=10):
    log_var = torch.clamp(log_var, -10, 10).squeeze()  # Squeeze to scalar
    std = torch.exp(0.5 * log_var)
    logits_exp = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_exp)
    corrupted = logits_exp + noise * std
    probs = F.softmax(corrupted, dim=-1)
    avg_probs = probs.mean(dim=0)
    p_t = avg_probs[range(len(targets)), targets]
    focal_weight = (1 - p_t) ** gamma
    loss = (focal_weight * (-torch.log(p_t + 1e-8))).mean()
    return loss + 0.5 * log_var

def mc_bce_loss(logits, log_var, targets, T=10):
    log_var = torch.clamp(log_var, -10, 10).squeeze()  # Squeeze to scalar
    std = torch.exp(0.5 * log_var)
    logits_exp = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_exp)
    corrupted = logits_exp + noise * std
    probs = torch.sigmoid(corrupted)
    avg_probs = probs.mean(dim=0)
    return F.binary_cross_entropy(avg_probs, targets, reduction='mean') + 0.5 * log_var
'''
            cell['source'] = [new_source]
            print(f'Fixed loss functions in cell {i}')
            break

with open('notebooks/AURA_V8_Colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)

print('DONE: Added .squeeze() to log_var in both loss functions')
