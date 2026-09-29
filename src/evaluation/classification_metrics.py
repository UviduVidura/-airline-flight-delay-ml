"""Classification metrics for flight delay prediction."""

from __future__ import annotations

from sklearn.metrics import accuracy_score as sklearn_accuracy_score
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, log_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_proba=None, threshold: float = 0.5) -> dict:
    """Compute the standard binary classification metrics for this project."""
    metrics = {
        "accuracy": float(sklearn_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": float(threshold),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=[0, 1]))
        except ValueError:
            metrics["log_loss"] = None
    return metrics
