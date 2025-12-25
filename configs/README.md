# Configs Directory

Complete configuration files documenting the experimental design, dataset, and model architectures used in this study.

## Structure

```
configs/
├── experiment/
│   └── factorial_cv.yaml       # 2×2 Factorial design (4 configs × 3 folds)
├── data/
│   └── visium_hd_crc.yaml      # Visium HD CRC dataset specification
└── model/
    ├── hist2st_decoder.yaml    # Hist2ST decoder (best performer)
    └── img2st_decoder.yaml     # Img2ST decoder (U-Net baseline)
```

## Experiment Configuration

**`experiment/factorial_cv.yaml`** - Complete experimental design

- **Design**: 2×2 factorial (Loss × Decoder)
- **Factors**:
  - Loss: {MSE, Poisson NLL}
  - Decoder: {Hist2ST, Img2ST}
- **CV Strategy**: 3-fold leave-one-patient-out
- **Models**:
  - E': Hist2ST + Poisson (BEST, SSIM=0.542)
  - F: Img2ST + Poisson (rank 2)
  - D': Hist2ST + MSE (rank 3)
  - G: Img2ST + MSE (WORST, SSIM=0.142)

### Cross-Validation Folds

| Fold | Train Patients | Test Patient |
|------|----------------|--------------|
| 1 | P1, P2 | P5 |
| 2 | P1, P5 | P2 |
| 3 | P2, P5 | P1 |

## Data Configuration

**`data/visium_hd_crc.yaml`** - Dataset specification

- **Source**: 10x Genomics Visium HD
- **Tissue**: Colorectal cancer (CRC)
- **Resolution**: 2μm bins
- **Patients**: 3 (P1, P2, P5)
- **Genes**: 50 (top variance)
- **Sparsity**: ~95% zeros (extreme!)
- **Preprocessing**: **Raw UMI counts, NO normalization** (critical for Poisson)

### Gene Categories (7 categories, 50 genes)

1. **Epithelial** (8): CEACAM5, EPCAM, KRT8, etc.
2. **Immune** (7): JCHAIN, IGHA1, CD74, etc.
3. **Secretory** (4): MUC12, PIGR, FCGBP
4. **Stromal** (5): VIM, COL1A1, COL3A1, DES
5. **Housekeeping** (6): TMSB10, ACTB, TMSB4X
6. **Mitochondrial** (11): MT-ND5, MT-CO2, MT-CYB, etc.
7. **Other** (9): TSPAN8, FTH1, PYGB, etc.

## Model Configurations

### `model/hist2st_decoder.yaml` - Best Performer

**Architecture**: Multi-pathway decoder with 3 parallel branches
- **CNN pathway**: Local texture (3×3 convolutions)
- **Transformer pathway**: Global context (8-head attention)
- **GNN pathway**: Neighborhood structure (k=8 neighbors)
- **Fusion**: Concatenate + upsample to 128×128

**Key Features**:
- Encoder: Virchow2 (1024-dim, frozen)
- Output: Linear → exp() for Poisson
- **Bias init**: -3.0 (critical! exp(-3)≈0.05 matches ~95% sparsity)

**Performance**:
- E' (Hist2ST + Poisson): SSIM=0.542 ± 0.019 ⭐ BEST
- D' (Hist2ST + MSE): SSIM=0.200 ± 0.012
- **Improvement**: 2.7× (p<0.001)

### `model/img2st_decoder.yaml` - U-Net Baseline

**Architecture**: Simple U-Net with skip connections
- Standard encoder-decoder with 4 downsampling levels
- Skip connections via concatenation
- No transformer or GNN components

**Key Features**:
- Encoder: Virchow2 (same as Hist2ST)
- Output: Linear → exp() for Poisson
- **Bias init**: -3.0 (same critical initialization)

**Performance**:
- F (Img2ST + Poisson): SSIM=0.268 ± 0.013
- G (Img2ST + MSE): SSIM=0.142 ± 0.007
- **Improvement**: 1.9× (Poisson still helps!)

## Critical Training Details

### Bias Initialization

**Most important hyperparameter** for Poisson training on sparse data:

```python
# Output layer bias initialized to -3.0
output_bias = -3.0
# exp(-3.0) ≈ 0.05 ≈ fraction of non-zero bins
```

Without this initialization, Poisson training fails to converge!

### Loss Functions

**Poisson NLL** (recommended):
```
L = lambda - k * log(lambda)
where lambda = exp(model_output)
```

**MSE** (fails on sparse data):
```
L = (prediction - target)^2
Problem: predicting 0 everywhere minimizes loss!
```

### Data Preprocessing

**CRITICAL**: Use raw UMI counts, NO normalization!
- Log-normalization breaks Poisson likelihood
- CPM/TPM normalization breaks count-based model
- Keep raw integer counts for Poisson NLL

## Usage

These configs document the exact experimental setup. Training script usage:

```bash
# Train E' (best model)
python scripts/train_2um_cv_3fold.py --decoder hist2st --loss poisson --fold all

# Train all 4 models × 3 folds = 12 runs
for decoder in hist2st img2st; do
  for loss in poisson mse; do
    python scripts/train_2um_cv_3fold.py --decoder $decoder --loss $loss --fold all
  done
done
```

## Results Summary

From `tables/table_s1_pergene_metrics.csv`:

| Model | Decoder | Loss | SSIM (mean ± std) | Rank |
|-------|---------|------|-------------------|------|
| E' | Hist2ST | Poisson | **0.542 ± 0.019** | 1st ⭐ |
| F | Img2ST | Poisson | 0.268 ± 0.013 | 2nd |
| D' | Hist2ST | MSE | 0.200 ± 0.012 | 3rd |
| G | Img2ST | MSE | 0.142 ± 0.007 | 4th |

**Key Finding**: Poisson loss is essential at 2μm resolution. All 50 genes benefit (100% improvement rate).
