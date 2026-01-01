from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def _read_tabular(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in [".parquet"]:
        return pd.read_parquet(path)
    if suf in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_id = (root / "models" / "registry" / "latest.txt").read_text(encoding="utf-8").strip()

    tables_dir = root / "models" / "runs" / run_id / "tables"

    matches = sorted(tables_dir.glob("holdout_predictions.*"))
    if not matches:
        raise FileNotFoundError(f"No holdout_predictions.* found in: {tables_dir}")

    preds_path = matches[0]
    df = _read_tabular(preds_path)

    is_classification = ("score" in df.columns) and ("prediction" in df.columns)

    if is_classification:
        target_col = "is_high_value" if "is_high_value" in df.columns else df.columns[-1]

        y_true = df[target_col].astype(int)
        y_pred = df["prediction"].astype(int)
        y_score = df["score"].astype(float)

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve — Model Fit (Holdout)")
        plt.legend()
        plt.show()

        # Score distribution
        plt.figure()
        plt.hist(y_score, bins=20)
        plt.xlabel("Score (P(y=1))")
        plt.ylabel("Count")
        plt.title("Score Distribution — Holdout")
        plt.show()

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Confusion Matrix — Holdout")
        plt.show()

    else:
       
        if "prediction" not in df.columns:
            raise KeyError("Regression expects a 'prediction' column in holdout_predictions.*")

        candidate_targets = [c for c in df.columns if c != "prediction"]
        target_col = candidate_targets[-1]

        y_true = df[target_col].astype(float)
        y_pred = df["prediction"].astype(float)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)

        # 1) Scatter: True vs Pred
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title(f"Regression Fit — Holdout (MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f})")
        plt.show()

        # 2) Residuals histogram
        residuals = y_true - y_pred
        plt.figure()
        plt.hist(residuals, bins=30)
        plt.xlabel("Residual (y_true - y_pred)")
        plt.ylabel("Count")
        plt.title("Residuals Distribution — Holdout")
        plt.show()

if __name__ == "__main__":
    main()
