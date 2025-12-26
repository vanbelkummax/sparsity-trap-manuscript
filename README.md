# The Sparsity Trap: MSE vs Poisson for 2μm Spatial Transcriptomics

**Publication-ready manuscript with multi-scale validation and corrected SSIM implementation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-success)](tests/)
[![SSIM](https://img.shields.io/badge/SSIM%20bug-fixed-green)](METRICS_CLARIFICATION.md)

> **Full repository with all 50 genes**: https://github.com/vanbelkummax/sparsity-trap-publication

---

## Important Update (December 25, 2025)

**SSIM Bug Fixed**: A code review identified that `src/evaluation/metrics.py` incorrectly returned Pearson correlation when masks were applied. Investigation confirmed that the **training scripts used correct SSIM computation**, so all manuscript results remain valid. The bug has been fixed for code consistency. See [METRICS_CLARIFICATION.md](METRICS_CLARIFICATION.md) for details.

---

## Key Finding

**MSE loss collapses on 2μm data. Poisson loss recovers structure with 2.7× SSIM improvement (p<0.001). All 50 genes benefit.**

![Combined Figure 1](figures/manuscript/figure_1_combined.png)

---

## Main Results

| Model | Decoder | Loss | SSIM 2μm | Rank |
|-------|---------|------|----------|------|
| **E'** | **Hist2ST** | **Poisson** | **0.542 ± 0.019** | 1st |
| F | Img2ST | Poisson | 0.268 ± 0.013 | 2nd |
| D' | Hist2ST | MSE | 0.200 ± 0.012 | 3rd |
| G | Img2ST | MSE | 0.142 ± 0.007 | 4th |

**Key Metrics**:
- SSIM Improvement: **2.7×** (p<0.001)
- Genes Benefiting: **50/50** (100%)
- Mean Δ-SSIM: **+0.412**
- Sparsity Correlation: **r=0.577**, p<0.0001

---

## Visual Examples

### Epithelial Markers

| CEACAM5 (+0.70) | EPCAM |
|:---:|:---:|
| ![CEACAM5](figures/wsi/CEACAM5_2um_WSI_improved.png) | ![EPCAM](figures/wsi/EPCAM_2um_WSI_improved.png) |

| KRT8 (+0.67) | MUC12 (Secretory, +0.63) |
|:---:|:---:|
| ![KRT8](figures/wsi/KRT8_2um_WSI_improved.png) | ![MUC12](figures/wsi/MUC12_2um_WSI_improved.png) |

### Top Gene: TSPAN8 (+0.73)

![TSPAN8](figures/wsi/TSPAN8_2um_WSI_improved.png)

### Other Categories

| JCHAIN (Immune, +0.46) | VIM (Stromal, +0.18) |
|:---:|:---:|
| ![JCHAIN](figures/wsi/JCHAIN_2um_WSI_improved.png) | ![VIM](figures/wsi/VIM_2um_WSI_improved.png) |

---

## Repository Contents

**Size**: ~65MB (vs 721MB full repo)

**Manuscript**:
- ✅ **sparsity_trap_manuscript_v2.tex** - Final publication-ready manuscript (12 pages)
- ✅ **sparsity_trap_manuscript_v2.pdf** - Compiled PDF with embedded figures (20 MB)
- ✅ All 24 citations verified and relevant

**Figures**:
- ✅ 5 main figures (factorial, category, main effects, representative genes, multi-scale)
- ✅ 7 WSI examples (whole slide images showing tissue-level validation)
- ✅ 2 granular glandular examples (2μm subcellular architecture)
- ✅ Combined multi-panel Figure 5 with WSI + glandular detail

**Code & Data**:
- ✅ Complete source code (SSIM bug fixed)
- ✅ 17 tests (100% passing)
- ✅ Training & visualization scripts
- ✅ Per-gene metrics (all 50 genes, CSV)
- ✅ Experiment configs
- ✅ **METRICS_CLARIFICATION.md** - Documentation of SSIM computation

**Representative Genes** (1 per category):
- TSPAN8 (Other, +0.73)
- CEACAM5 (Epithelial, +0.70)
- EPCAM (Epithelial)
- KRT8 (Epithelial, +0.67)
- MUC12 (Secretory, +0.63)
- JCHAIN (Immune, +0.46)
- VIM (Stromal, +0.18)

---

## Installation

```bash
git clone https://github.com/vanbelkummax/sparsity-trap-manuscript.git
cd sparsity-trap-manuscript
pip install -e .
pytest tests/  # 17/17 should pass
```

**Requirements**: Python 3.10+, PyTorch 2.0+, 24GB+ VRAM

---

## Usage

```bash
# Train best model
python scripts/train_2um_cv_3fold.py --decoder hist2st --loss poisson --fold all

# Generate figures
python scripts/create_manuscript_figures.py
```

---

## Citation

```bibtex
@software{vanbelkum2025sparsity_trap,
  author = {Van Belkum, Max},
  title = {The Sparsity Trap: Why MSE Fails and Poisson Succeeds for 2μm Spatial Transcriptomics},
  year = {2025},
  url = {https://github.com/vanbelkummax/sparsity-trap-manuscript}
}
```

---

## License & Contact

MIT License • max.vanbelkum@vanderbilt.edu • [@vanbelkummax](https://github.com/vanbelkummax)

**For all 50 genes**: https://github.com/vanbelkummax/sparsity-trap-publication
