"""Integration test: Train for 1 epoch and verify loss decreases."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.losses import PoissonNLLLoss, MSELoss


class DummyModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, in_channels=3, out_channels=50):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return torch.exp(self.conv2(x))  # Positive output for Poisson


class TestIntegration:
    def test_poisson_loss_training(self):
        """Test that training with Poisson loss decreases loss."""
        model = DummyModel()
        loss_fn = PoissonNLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dummy data
        batch_size, h, w, genes = 2, 128, 128, 50
        images = torch.rand(batch_size, 3, h, w)
        targets = torch.rand(batch_size, genes, h, w) * 10

        # Initial loss
        model.train()
        pred_init = model(images)
        loss_init = loss_fn(pred_init, targets)

        # Train for 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

        # Final loss
        pred_final = model(images)
        loss_final = loss_fn(pred_final, targets)

        # Loss should decrease
        assert loss_final < loss_init, f"Loss did not decrease: {loss_init:.4f} -> {loss_final:.4f}"

    def test_mse_loss_training(self):
        """Test that training with MSE loss decreases loss."""
        model = DummyModel()
        loss_fn = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Dummy data
        batch_size, h, w, genes = 2, 128, 128, 50
        images = torch.rand(batch_size, 3, h, w)
        targets = torch.rand(batch_size, genes, h, w) * 10

        # Initial loss
        model.train()
        pred_init = model(images)
        loss_init = loss_fn(pred_init, targets)

        # Train for 10 steps
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(images)
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

        # Final loss
        pred_final = model(images)
        loss_final = loss_fn(pred_final, targets)

        # Loss should decrease
        assert loss_final < loss_init, f"Loss did not decrease: {loss_init:.4f} -> {loss_final:.4f}"
