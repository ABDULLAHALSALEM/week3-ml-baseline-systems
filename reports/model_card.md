# Model Card — Week 3 Baseline

## Problem
- **Target**: `is_high_value` (binary classification: high-value customer = 1).
- **Unit of analysis**: Individual user (`user_id`).
- **Decision enabled**: Identify high-value users to prioritize marketing or retention actions.

## Data
- **Feature table**: `data/processed/features.parquet`
- **Features used**:
  - `country`
  - `n_orders`
  - `avg_amount`
  - `total_amount`
- **Dataset hash (sha256)**:  
  `bce383b3c7a6501454b05be84208fa2880d2f005f186ca71f54da25fcd5b6aed` :contentReference[oaicite:0]{index=0}

## Splits
- **Holdout strategy**: Random split
- **Test size**: 20%
- **Seed**: 42
- **Positive rate (holdout)**: 0.20 :contentReference[oaicite:1]{index=1}

## Metrics (holdout)
- **Baseline (fixed threshold = 0.5)**:
  - ROC AUC: 0.50
  - PR AUC: 0.20
  - Accuracy: 0.80
  - Precision: 0.00
  - Recall: 0.00
  - F1: 0.00 :contentReference[oaicite:2]{index=2}

- **Model (Logistic Regression)**:
  - ROC AUC: 1.00
  - PR AUC: 1.00
  - Accuracy: 1.00
  - Precision: 1.00
  - Recall: 1.00
  - F1: 1.00
  - ROC AUC 95% CI: [1.00, 1.00] :contentReference[oaicite:3]{index=3}

## Limitations
- Metrics are computed on a small synthetic dataset and may not generalize to real-world data.
- Perfect scores suggest potential simplicity or separability in the sample data.
- No temporal or group-based split was applied.

## Monitoring sketch
- Track prediction score distribution over time.
- Monitor positive rate drift compared to training holdout (0.20).
- Periodically recompute ROC AUC and precision/recall on fresh labeled data.

## Reproducibility
- **Run id**: `2026-01-01T15-12-57Z__classification__seed42` :contentReference[oaicite:4]{index=4}
- **Git commit**: recorded in `run_meta.json`
- **Environment**: `models/runs/2026-01-01T15-12-57Z__classification__seed42/env/pip_freeze.txt`
