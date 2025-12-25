# Scripts Directory

Training, evaluation, and visualization scripts for the publication.

## Training Scripts

- **train_factorial.py** - Train models with factorial design (4 configs × 3 folds)
- **reproduce_paper.sh** - One-click reproduction of all results

## Evaluation Scripts

- **evaluate_model.py** - Evaluate trained models on test sets
- **generate_figures.py** - Generate all publication figures

## Visualization Scripts

- **create_manuscript_figures.py** - Generate main figures (1-4, S1)
- **create_wsi_and_tiles.py** - Generate WSI and tile comparisons

## Usage

### Train all models (3-fold CV)
```bash
python scripts/train_factorial.py --decoder hist2st --loss poisson --fold all
```

### Evaluate models
```bash
python scripts/evaluate_model.py --fold all
```

### Generate figures
```bash
python scripts/generate_figures.py
```

### One-click reproduction
```bash
bash scripts/reproduce_paper.sh
```
