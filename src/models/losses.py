import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: class indices (for CE) or one-hot
        # Assuming BCEWithLogits for multi-label emotion or CE for single
        
        # Implementation for CrossEntropy (multi-class, single label per sample for now)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super(UncertaintyLoss, self).__init__()
        # sigma parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros((num_tasks)))

    def forward(self, loss_toxicity, loss_emotion):
        # L = 1/2sigma1^2 * L1 + log(sigma1) + ...
        
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * loss_toxicity + self.log_vars[0]
        
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * loss_emotion + self.log_vars[1]
        
        return loss1 + loss2
