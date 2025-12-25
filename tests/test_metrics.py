import pytest
import torch
import numpy as np
from src.evaluation.metrics import compute_ssim, compute_pcc


class TestSSIM:
    def test_ssim_identical_images(self):
        """Test SSIM between identical images."""
        img1 = torch.rand(128, 128)
        img2 = img1.clone()

        ssim = compute_ssim(img1, img2)

        assert np.isclose(ssim, 1.0, atol=1e-5)

    def test_ssim_different_images(self):
        """Test SSIM between different images."""
        img1 = torch.zeros(128, 128)
        img2 = torch.ones(128, 128)

        ssim = compute_ssim(img1, img2)

        assert 0.0 <= ssim < 1.0

    def test_ssim_with_mask(self):
        """Test SSIM with tissue mask."""
        img1 = torch.rand(128, 128)
        img2 = torch.rand(128, 128)
        mask = torch.ones(128, 128)
        mask[:64, :64] = 0  # Mask out top-left quadrant

        ssim_full = compute_ssim(img1, img2)
        ssim_masked = compute_ssim(img1, img2, mask=mask)

        # Masked SSIM should only consider unmasked region
        assert ssim_masked != ssim_full

    def test_ssim_batched(self):
        """Test SSIM with batched inputs."""
        batch_size, h, w = 4, 128, 128
        img1 = torch.rand(batch_size, h, w)
        img2 = torch.rand(batch_size, h, w)

        ssims = compute_ssim(img1, img2, batched=True)

        assert len(ssims) == batch_size
        # SSIM can range from -1 to 1
        assert all(-1.0 <= s <= 1.0 for s in ssims)
        assert all(isinstance(s, float) for s in ssims)


class TestPCC:
    def test_pcc_perfect_correlation(self):
        """Test PCC with perfect positive correlation."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])

        pcc = compute_pcc(pred, target)

        assert np.isclose(pcc, 1.0, atol=1e-5)

    def test_pcc_perfect_anti_correlation(self):
        """Test PCC with perfect negative correlation."""
        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([4.0, 3.0, 2.0, 1.0])

        pcc = compute_pcc(pred, target)

        assert np.isclose(pcc, -1.0, atol=1e-5)

    def test_pcc_zero_correlation(self):
        """Test PCC with zero correlation."""
        pred = torch.tensor([1.0, 2.0, 1.0, 2.0])
        target = torch.tensor([1.0, 1.0, 2.0, 2.0])

        pcc = compute_pcc(pred, target)

        assert -0.5 < pcc < 0.5  # Near zero

    def test_pcc_with_mask(self):
        """Test PCC with tissue mask."""
        pred = torch.rand(128, 128)
        target = torch.rand(128, 128)
        mask = torch.ones(128, 128)
        mask[:64, :64] = 0  # Mask out top-left quadrant

        pcc_full = compute_pcc(pred, target)
        pcc_masked = compute_pcc(pred, target, mask=mask)

        # Masked PCC should only consider unmasked region
        assert pcc_masked != pcc_full
