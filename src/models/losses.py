"""Loss functions for spatial transcriptomics prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.

    For count data y ~ Poisson(lambda), the NLL is:
        L = lambda - y * log(lambda)

    This loss is critical for sparse spatial transcriptomics because:
    - When y > 0, predicting lambda -> 0 gives INFINITE loss (prevents "gray fog")
    - When y = 0, loss = lambda (encourages small predictions)
    - Naturally handles mean-variance relationship (Var = Mean)

    Args:
        eps: Small constant to prevent log(0). Default: 1e-8
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, eps=1e-8, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted lambda values (B, C, H, W). Must be positive.
            targets: Ground truth counts (B, C, H, W). Non-negative integers.

        Returns:
            Scalar loss if reduction='mean', otherwise per-element losses.
        """
        # Clamp predictions to prevent log(0)
        predictions = torch.clamp(predictions, min=self.eps)

        # Poisson NLL: lambda - k * log(lambda)
        loss = predictions - targets * torch.log(predictions)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss.

    Standard L2 loss: (prediction - target)^2

    WARNING: For sparse count data (95% zeros), MSE encourages predicting
    zero everywhere because that minimizes the loss. This is the "sparsity trap".
    Use PoissonNLLLoss instead.

    Args:
        reduction: 'mean', 'sum', or 'none'. Default: 'mean'
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Predicted values (B, C, H, W)
            targets: Ground truth values (B, C, H, W)

        Returns:
            Scalar loss if reduction='mean', otherwise per-element losses.
        """
        loss = (predictions - targets) ** 2

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_type='poisson'):
    """
    Factory function to get loss by name.

    Args:
        loss_type: 'poisson' or 'mse'

    Returns:
        Loss function instance
    """
    if loss_type == 'poisson':
        return PoissonNLLLoss()
    elif loss_type == 'mse':
        return MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Use 'poisson' or 'mse'.")
