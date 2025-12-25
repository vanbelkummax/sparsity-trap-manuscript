# Data Documentation

## Dataset: 10x Genomics Visium HD CRC

### Overview
This study uses publicly available Visium HD spatial transcriptomics data from colorectal cancer (CRC) tissue sections.

- **Technology**: 10x Genomics Visium HD
- **Resolution**: 2μm bins (ultra-high resolution)
- **Tissue Type**: Human colorectal adenocarcinoma
- **Source**: 10x Genomics public datasets

### Patients
Three CRC patient samples were used for 3-fold cross-validation:

| Patient ID | Description | Use in CV |
|------------|-------------|-----------|
| P1 | CRC tissue section | Train/Test (Folds 2,3 train; Fold 3 test) |
| P2 | CRC tissue section | Train/Test (Folds 1,2 train; Fold 2 test) |
| P5 | CRC tissue section | Train/Test (Folds 1,3 train; Fold 1 test) |

### Preprocessing

**Critical**: Data is used as **raw UMI counts** with **NO normalization**. This is essential for Poisson loss training.

1. **Gene Selection**: Top 50 genes by variance across all patients
2. **Patch Extraction**:
   - Patch size: 128×128 bins at 2μm resolution
   - Corresponding H&E images: 512×512 pixels (4× downsampled to 128×128 for encoder)
3. **Quality Control**:
   - Tissue detection masks ensure patches contain actual tissue
   - No smoothing or imputation applied

### Sparsity Statistics

At 2μm resolution, the data is **extremely sparse**:

| Patient | Non-zero Fraction | Sparsity (% zeros) |
|---------|-------------------|-------------------|
| P1 | 3.4% | 96.6% |
| P2 | 5.2% | 94.8% |
| P5 | 6.1% | 93.9% |
| **Mean** | **~5%** | **~95%** |

This extreme sparsity is the root cause of the "sparsity trap" where MSE loss fails catastrophically.

### Gene Categories

The 50 genes span 7 functional categories:

| Category | Count | Examples | Mean Sparsity |
|----------|-------|----------|---------------|
| **Epithelial** | 8 | CEACAM5, EPCAM, KRT8, CEACAM6 | 95.3% |
| **Immune** | 7 | JCHAIN, IGHA1, CD74, IGHG1 | 93.1% |
| **Secretory** | 4 | MUC12, PIGR, FCGBP | 94.2% |
| **Stromal** | 5 | VIM, COL1A1, COL3A1, DES | 93.8% |
| **Housekeeping** | 6 | TMSB10, ACTB, TMSB4X | 91.5% |
| **Mitochondrial** | 11 | MT-ND5, MT-CO2, MT-CYB | 78.4% |
| **Other** | 9 | TSPAN8, FTH1, PYGB | 95.7% |

### Data Files

The actual data files are stored at: `/home/user/visium-hd-2um-benchmark/`

Structure:
```
visium-hd-2um-benchmark/
├── patches/
│   ├── P1/
│   │   ├── images/          # H&E patches (512×512)
│   │   ├── labels_2um/      # Gene expression at 2μm (128×128×50)
│   │   ├── labels_8um/      # Gene expression at 8μm (32×32×50)
│   │   └── masks/           # Tissue masks (128×128)
│   ├── P2/
│   └── P5/
└── gene_metadata.csv        # Gene names and categories
```

### Data Loading

The training script (`scripts/train_2um_cv_3fold.py`) automatically loads data for the specified CV fold:

```python
# Fold 1 example
train_patients = ['P1', 'P2']  # 80% train, 20% val split
test_patient = 'P5'            # Hold-out test
```

Each patch contains:
- **image**: H&E histology (3×512×512) → resized to 3×128×128
- **label_2um**: Gene expression (50×128×128), raw UMI counts
- **label_8um**: Gene expression (50×32×32), raw UMI counts
- **mask_2um**: Tissue detection mask (128×128), binary

### Data Augmentation

Training uses geometric augmentations applied jointly to image, labels, and mask:
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)
- 90° rotation (p=0.5)

**Critical**: Augmentations are applied to ALL modalities (image + labels + mask) to maintain spatial correspondence.

### Cross-Validation Strategy

**3-Fold Cross-Validation** ensures robust generalization:

| Fold | Train Patients | Val Patients | Test Patient |
|------|----------------|--------------|--------------|
| 1 | P1, P2 | P1, P2 (20%) | P5 |
| 2 | P1, P5 | P1, P5 (20%) | P2 |
| 3 | P2, P5 | P2, P5 (20%) | P1 |

Each patient serves as the hold-out test set exactly once, providing 3 independent performance estimates.

### Expected Results

From `tables/table_s1_pergene_metrics.csv`, you can see:
- All 50 genes analyzed
- Per-gene sparsity, SSIM, and PCC metrics
- MSE vs Poisson comparison for each gene

**Key Finding**: 50/50 genes (100%) show improved SSIM with Poisson loss, with mean improvement of +0.412 SSIM (2.7× better than MSE).

## Downloading the Data

The original Visium HD CRC dataset is available from 10x Genomics:
- https://www.10xgenomics.com/products/visium-hd-spatial-gene-expression
- Search for "Colorectal Cancer" or "CRC" datasets

**Note**: This publication package contains all pre-computed results, so data download is NOT required to reproduce the figures and tables. Data download is only needed if you want to re-train models from scratch.

## Data Usage Terms

Please cite the original 10x Genomics Visium HD publication when using this data:
- Dataset provided by 10x Genomics under their standard terms of use
- See 10x Genomics website for the most current citation information
