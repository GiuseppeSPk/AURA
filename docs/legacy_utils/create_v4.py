import json

# Read V3 notebook
with open('notebooks/AURA_Bayesian_V3_CleanData.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title cell
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and 'AURA V3' in ''.join(cell['source']):
        cell['source'] = [
            "# AURA V4: BERT + Clean Data + Focal Loss\n",
            "\n",
            "---\n",
            "## PRIMA DI ESEGUIRE:\n",
            "1. **Settings** -> **Accelerator** -> **GPU T4 x2**\n",
            "2. **Add Input** -> Carica `aura-data-v2`\n",
            "---\n",
            "\n",
            "### V4 Features\n",
            "| Component | Implementation |\n",
            "|-----------|----------------|\n",
            "| Backbone | BERT-base |\n",
            "| Data | Clean (7 emotion classes) |\n",
            "| Loss (Toxicity) | **Focal Loss** (γ=2.0) |\n",
            "| Loss (Emotions) | BCE |\n",
            "| MTL Balancing | Kendall Uncertainty |\n",
            "\n",
            "**Theoretical Advantage**: Focal Loss dynamically down-weights easy examples, focusing on hard negatives."
        ]
        break

# Update loss function
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Replace the monte_carlo_uncertainty_loss_classification function
        if 'def monte_carlo_uncertainty_loss_classification' in source:
            new_source = """def focal_loss_with_uncertainty(logits, log_var, targets, gamma=2.0, T=10):
    \"\"\"
    Focal Loss integrated with Kendall's Uncertainty (Lin et al., 2017 + Kendall et al., 2018).
    
    Focal Loss: FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    where p_t is the probability of the correct class.
    
    Args:
        logits: Model predictions [batch, num_classes]
        log_var: Log-variance parameter (Kendall)
        targets: Ground truth labels [batch]
        gamma: Focal loss focusing parameter (default: 2.0)
        T: Monte Carlo samples
    \"\"\"
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped)
    
    # Monte Carlo Sampling
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_expanded)
    corrupted_logits = logits_expanded + (noise * std)
    
    # Average probabilities
    probs = F.softmax(corrupted_logits, dim=-1)
    avg_probs = torch.mean(probs, dim=0)
    
    # Get probabilities of correct class (p_t)
    p_t = avg_probs[range(len(targets)), targets]
    
    # Focal Loss formula: -(1 - p_t)^gamma * log(p_t)
    focal_weight = (1 - p_t) ** gamma
    ce_loss = -torch.log(p_t + 1e-8)
    focal_loss = (focal_weight * ce_loss).mean()
    
    # Kendall regularization
    regularization = 0.5 * log_var_clamped
    
    return focal_loss + regularization


def monte_carlo_uncertainty_loss_multilabel(logits, log_var, targets, T=10):
    \"\"\"
    Bayesian Uncertainty Loss for Multi-Label (Emotions).
    \"\"\"
    log_var_clamped = torch.clamp(log_var, min=-10, max=10)
    std = torch.exp(0.5 * log_var_clamped)
    
    logits_expanded = logits.unsqueeze(0).expand(T, -1, -1)
    noise = torch.randn_like(logits_expanded)
    corrupted_logits = logits_expanded + (noise * std)
    
    probs = torch.sigmoid(corrupted_logits)
    avg_probs = torch.mean(probs, dim=0)
    
    bce = F.binary_cross_entropy(avg_probs, targets, reduction='mean')
    regularization = 0.5 * log_var_clamped
    
    return bce + regularization"""
            
            cell['source'] = [new_source]
            break

# Update training loop to use focal loss
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        if 'monte_carlo_uncertainty_loss_classification(' in source and 'tox_loss =' in source:
            # Replace the function call
            new_source = source.replace(
                'monte_carlo_uncertainty_loss_classification(\n                tox_logits[tox_mask], \n                tox_log_var, \n                tox_targets[tox_mask],\n                class_weights,\n                T=config[\'mc_samples\']\n            )',
                'focal_loss_with_uncertainty(\n                tox_logits[tox_mask], \n                tox_log_var, \n                tox_targets[tox_mask],\n                gamma=2.0,\n                T=config[\'mc_samples\']\n            )'
            )
            cell['source'] = [new_source]

# Remove CLASS_WEIGHTS from config
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'CLASS_WEIGHTS' in source and 'torch.tensor' in source:
            # Comment out class weights
            new_source = source.replace(
                "# CLASS WEIGHTS - Calcolati da OLID train: NOT=7975, OFF=3941\n# Formula: weight = total / (2 * class_count), normalizzato\nCLASS_WEIGHTS = torch.tensor([1.0, 2.02]).to(device)\nprint(f\"Class Weights: NOT=1.0, OFF=2.02\")",
                "# V4: Using Focal Loss instead of Class Weights\n# Focal Loss (gamma=2.0) automatically focuses on hard examples\nprint(f\"V4: Focal Loss enabled (gamma=2.0)\")"
            )
            cell['source'] = [new_source]

# Update title in training loop
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'STARTING V3 TRAINING' in source:
            new_source = source.replace(
                'STARTING V3 TRAINING (Clean Data + Class Weights)',
                'STARTING V4 TRAINING (Clean Data + Focal Loss)'
            ).replace(
                'aura_v3_best.pt',
                'aura_v4_focal_best.pt'
            )
            cell['source'] = [new_source]

# Save V4 notebook
with open('notebooks/AURA_Bayesian_V4_FocalLoss.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print("✅ V4 Notebook created!")
print("   - Replaced Class Weights with Focal Loss (gamma=2.0)")
print("   - Integrated with Kendall's uncertainty framework")
print("   - Model: BERT + Clean Data + Focal Loss")
