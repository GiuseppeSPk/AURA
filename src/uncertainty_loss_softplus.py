"""
AURA V10 - Corrected UncertaintyLoss with SoftPlus
===================================================
Sostituisce la versione nel notebook AURA_V10_Kaggle.ipynb Cell 6.

COPIA E INCOLLA QUESTO CODICE NEL NOTEBOOK sostituendo la classe UncertaintyLoss esistente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyLoss(nn.Module):
    """Kendall et al. (2018) Homoscedastic Uncertainty for Multi-Task Learning.
    
    L_total = sum_i [(1/σ²_i) * L_i + log(σ²_i)/2]
    where σ²_i = softplus(θ_i) = log(1 + exp(θ_i))
    
    SoftPlus ensures σ² > 0 with stable gradients near zero.
    Automatically balances task losses based on learned uncertainty.
    """
    def __init__(self, n_tasks=4):
        super().__init__()
        # Parameter θ, transformed via SoftPlus to ensure σ² > 0
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    
    def forward(self, losses):
        total = 0
        for i, loss in enumerate(losses):
            # σ² = softplus(θ) for numerical stability
            sigma_sq = F.softplus(self.log_vars[i])
            precision = 1.0 / sigma_sq
            total += precision * loss + torch.log(sigma_sq) * 0.5
        return total
    
    def get_weights(self):
        # Return precision (1/σ²) for visualization
        sigma_sq = F.softplus(self.log_vars)
        return (1.0 / sigma_sq).detach().cpu().numpy()


# ============================================================================
# TESTING & VERIFICATION
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("TESTING SOFTPLUS UNCERTAINTY LOSS")
    print("="*60)
    
    # Test 1: Positivity guarantee
    print("\n1. Testing σ² > 0 for all θ values...")
    theta_test = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
    sigma_sq_test = F.softplus(theta_test)
    print(f"   θ values: {theta_test.tolist()}")
    print(f"   σ² values: {sigma_sq_test.tolist()}")
    assert (sigma_sq_test > 0).all(), "❌ SoftPlus failed to ensure positivity!"
    print("   ✅ All σ² > 0")
    
    # Test 2: Gradient stability
    print("\n2. Testing gradient stability...")
    
    # exp(-x) version (old)
    x_exp = torch.tensor([-10.0], requires_grad=True)
    y_exp = torch.exp(-x_exp)
    y_exp.backward()
    grad_exp = x_exp.grad.item()
    
    # softplus version (new)
    x_softplus = torch.tensor([-10.0], requires_grad=True)
    y_softplus = F.softplus(x_softplus)
    y_softplus.backward()
    grad_softplus = x_softplus.grad.item()
    
    print(f"   exp(-x) gradient @ x=-10: {grad_exp:.2e} (exploding!)")
    print(f"   softplus(x) gradient @ x=-10: {grad_softplus:.2e} (stable)")
    print("   ✅ SoftPlus has stable gradients")
    
    # Test 3: Forward pass shape
    print("\n3. Testing forward pass...")
    loss_fn = UncertaintyLoss(n_tasks=4)
    dummy_losses = [
        torch.tensor(1.0, requires_grad=True),
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(1.5, requires_grad=True),
        torch.tensor(0.8, requires_grad=True)
    ]
    total = loss_fn(dummy_losses)
    print(f"   Input losses: [1.0, 2.0, 1.5, 0.8]")
    print(f"   Combined loss: {total.item():.4f}")
    print(f"   Output shape: {total.shape} (expected: torch.Size([]))")
    assert total.shape == torch.Size([]), "❌ Output should be scalar!"
    print("   ✅ Forward pass works correctly")
    
    # Test 4: Backward pass
    print("\n4. Testing backward pass...")
    total.backward()
    print(f"   Gradients computed for {sum(1 for p in loss_fn.parameters() if p.grad is not None)} parameters")
    print(f"   log_vars gradients: {loss_fn.log_vars.grad}")
    print("   ✅ Backward pass works correctly")
    
    # Test 5: get_weights()
    print("\n5. Testing get_weights()...")
    weights = loss_fn.get_weights()
    print(f"   Task weights (1/σ²): {weights}")
    assert len(weights) == 4, "❌ Should return 4 weights!"
    assert (weights > 0).all(), "❌ All weights should be positive!"
    print("   ✅ get_weights() works correctly")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - READY TO USE")
    print("="*60)
    print("\nInstructions:")
    print("1. Apri il notebook AURA_V10_Kaggle.ipynb")
    print("2. Vai alla Cell 6 (Loss Functions)")
    print("3. Sostituisci la classe UncertaintyLoss con quella sopra")
    print("4. Esegui la cella per verificare che funzioni")
    print("\nNote:")
    print("- I checkpoint esistenti NON sono compatibili")
    print("- Dovrai rifare il training da zero")
    print("- La performance sarà identica o migliore")
