# Evaluation Summary — Week 3 Baseline

## What was trained
- **Model family**: Logistic Regression (classification).
- **Preprocessing**:
  - Numerical features: median imputation.
  - Categorical features: most-frequent imputation + one-hot encoding.
  - Implemented via a single `Pipeline` to avoid training/serving skew.

## Results
- **Baseline (fixed threshold = 0.5)**:
  - ROC AUC: 0.50
  - PR AUC: 0.20
  - Accuracy: 0.80
  - Precision / Recall / F1: 0.00

- **Model (Logistic Regression)**:
  - ROC AUC: 1.00
  - PR AUC: 1.00
  - Accuracy: 1.00
  - Precision: 1.00
  - Recall: 1.00
  - F1: 1.00

The trained model significantly outperforms the baseline across all holdout metrics. 

## Error analysis
- The model achieves perfect performance on the holdout set, suggesting the dataset may be too small or linearly separable.
- No obvious label leakage was detected, but features such as `total_amount` are highly correlated with the target and should be reviewed carefully.
- Errors could emerge on real-world data with more noise, unseen categories, or distribution shift.

## Recommendation
- **Do not ship yet**.
- Although holdout performance is perfect, results are based on synthetic/sample data.
- Recommendation is to validate on a larger, more realistic dataset and introduce:
  - More diverse users,
  - Temporal splits,
  - Additional monitoring before production use.
