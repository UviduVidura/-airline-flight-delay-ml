"""Create simple evaluation visualizations from persisted metric files.

We generate:
- confusion matrix figure (from saved counts)
- metrics bar chart for classification/regression

No model retraining or prediction is performed.
"""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "model_results"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion(matrix, out_path: Path) -> None:
    cm = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(class_metrics: dict, reg_metrics: dict, out_path: Path) -> None:
    # select a small set of numeric metrics to show
    cls_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    cls_vals = [class_metrics.get(k, 0.0) for k in cls_keys]

    reg_keys = ["mae", "rmse", "r2"]
    reg_vals = [reg_metrics.get(k, 0.0) for k in reg_keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(cls_keys, cls_vals, color="C0")
    axes[0].set_title("Classification metrics")
    axes[0].set_ylim(0, 1)

    axes[1].bar(reg_keys, reg_vals, color="C1")
    axes[1].set_title("Regression metrics")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cls_file = METRICS_DIR / "classification_metrics.json"
    reg_file = METRICS_DIR / "regression_metrics.json"

    cls = json.loads(cls_file.read_text(encoding="utf-8")) if cls_file.exists() else {}
    reg = json.loads(reg_file.read_text(encoding="utf-8")) if reg_file.exists() else {}

    if "confusion_matrix" in cls:
        plot_confusion(cls["confusion_matrix"], OUT_DIR / "confusion_matrix.png")

    plot_metric_bars(cls, reg, OUT_DIR / "metrics_bar.png")
    print(f"Generated evaluation figures under {OUT_DIR}")


if __name__ == "__main__":
    main()
