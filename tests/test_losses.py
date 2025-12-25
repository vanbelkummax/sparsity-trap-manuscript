import pytest
import torch
from src.models.losses import PoissonNLLLoss, MSELoss


class TestPoissonNLLLoss:
    def test_poisson_nll_zero_target(self):
        """Test Poisson NLL when target is zero (k=0)."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])  # lambda values
        targets = torch.tensor([[0.0, 0.0, 0.0]])  # k=0 everywhere

        loss = loss_fn(predictions, targets)

        # L = lambda - k*log(lambda) = lambda when k=0
        expected = predictions.mean()  # Should just be mean of lambdas
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_poisson_nll_nonzero_target(self):
        """Test Poisson NLL when target is non-zero."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[2.0]])  # lambda=2
        targets = torch.tensor([[3.0]])  # k=3

        loss = loss_fn(predictions, targets)

        # L = lambda - k*log(lambda) = 2 - 3*log(2) = 2 - 3*0.693 = -0.079
        expected = 2.0 - 3.0 * torch.log(torch.tensor(2.0))
        assert torch.isclose(loss, expected, atol=1e-3)

    def test_poisson_nll_batch(self):
        """Test Poisson NLL with batched inputs."""
        loss_fn = PoissonNLLLoss()
        batch_size, h, w, genes = 4, 8, 8, 50
        predictions = torch.rand(batch_size, genes, h, w) * 10
        targets = torch.rand(batch_size, genes, h, w) * 20

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        # Poisson NLL can be negative when k*log(lambda) > lambda
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_poisson_nll_requires_positive_lambda(self):
        """Test that negative lambda predictions are handled."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[-1.0, 2.0]])  # Negative lambda
        targets = torch.tensor([[0.0, 3.0]])

        # Should either error or clamp to small positive value
        # Implementation should prevent log(0) or log(negative)
        loss = loss_fn(predictions, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMSELoss:
    def test_mse_zero_error(self):
        """Test MSE when prediction equals target."""
        loss_fn = MSELoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[1.0, 2.0, 3.0]])

        loss = loss_fn(predictions, targets)

        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_mse_simple(self):
        """Test MSE with known values."""
        loss_fn = MSELoss()
        predictions = torch.tensor([[2.0, 4.0]])
        targets = torch.tensor([[1.0, 3.0]])

        loss = loss_fn(predictions, targets)

        # MSE = mean((2-1)^2 + (4-3)^2) = mean(1 + 1) = 1.0
        expected = torch.tensor(1.0)
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_mse_batch(self):
        """Test MSE with batched inputs."""
        loss_fn = MSELoss()
        batch_size, h, w, genes = 4, 8, 8, 50
        predictions = torch.rand(batch_size, genes, h, w)
        targets = torch.rand(batch_size, genes, h, w)

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar
        assert loss >= 0
