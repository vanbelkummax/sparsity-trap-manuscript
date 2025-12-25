# Contributing to The Sparsity Trap

Thank you for your interest in contributing! This document provides guidelines for contributing to this publication package.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/vanbelkummax/sparsity-trap-publication.git
cd sparsity-trap-publication

# Create environment
conda env create -f environment.yml
conda activate sparsity-trap

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow existing code style
- Add tests for new features
- Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_losses.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### 4. Commit Changes

Use descriptive commit messages:

```bash
git commit -m "feat: add new loss function for zero-inflated models"
git commit -m "fix: correct SSIM calculation with mask"
git commit -m "docs: update REPRODUCTION.md with new steps"
```

Commit prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `style:` - Formatting changes

### 5. Submit Pull Request

- Ensure all tests pass
- Update documentation
- Reference any related issues

## Code Style

### Python

- Follow PEP 8
- Use type hints where appropriate
- Write docstrings for all public functions
- Maximum line length: 100 characters

Example:

```python
def compute_ssim(img1: torch.Tensor, img2: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Structural Similarity Index (SSIM).

    Args:
        img1: Predicted image (H, W)
        img2: Target image (H, W)
        mask: Optional binary mask (H, W)

    Returns:
        SSIM value in range [-1, 1]
    """
    # Implementation here
    pass
```

### Testing

- Write tests using pytest
- Use TDD: write tests first, then implementation
- Aim for >70% code coverage
- Test edge cases and error conditions

Example:

```python
class TestPoissonNLLLoss:
    def test_poisson_nll_zero_target(self):
        """Test Poisson NLL when target is zero."""
        loss_fn = PoissonNLLLoss()
        predictions = torch.tensor([[1.0, 2.0, 3.0]])
        targets = torch.tensor([[0.0, 0.0, 0.0]])

        loss = loss_fn(predictions, targets)

        expected = predictions.mean()
        assert torch.isclose(loss, expected, atol=1e-5)
```

## Areas for Contribution

### High Priority

1. **Zero-Inflated Loss Functions**
   - Implement ZIP (Zero-Inflated Poisson) loss
   - Implement ZINB (Zero-Inflated Negative Binomial) loss

2. **Additional Metrics**
   - Implement MS-SSIM (Multi-Scale SSIM)
   - Add gene-set enrichment metrics

3. **Visualization Tools**
   - Interactive figures with Plotly
   - Streamlit dashboard for results

### Medium Priority

4. **Data Loaders**
   - Support for other spatial transcriptomics platforms
   - Data augmentation strategies

5. **Model Architectures**
   - Alternative decoders (U-Net, ResNet-based)
   - Attention mechanisms

6. **Documentation**
   - Tutorial notebooks
   - API documentation with Sphinx

## Questions?

- Open an issue for bugs or feature requests
- Email: max.vanbelkum@vanderbilt.edu
- Check existing issues and pull requests first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
