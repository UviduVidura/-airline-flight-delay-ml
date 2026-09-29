"""Log existing model artifacts and metrics to a local MLflow tracking directory.

This script does NOT retrain models. It uses saved model artifacts and the
JSON metric files under `reports/model_results/` to create a reproducible
MLflow record of the trained models.
"""
from __future__ import annotations

import json
from pathlib import Path
import os

try:
    import mlflow
except Exception as exc:
    raise SystemExit("mlflow is required to run this script: install mlflow") from exc

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR}")

CLASS_PATH = ROOT / "models" / "classification_model.joblib"
REG_PATH = ROOT / "models" / "regression_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_results"

EXPERIMENT_NAME = "flight_delay_models"


def load_metrics() -> dict:
    out = {}
    cls = METRICS_PATH / "classification_metrics.json"
    reg = METRICS_PATH / "regression_metrics.json"
    if cls.exists():
        out["classification"] = json.loads(cls.read_text(encoding="utf-8"))
    if reg.exists():
        out["regression"] = json.loads(reg.read_text(encoding="utf-8"))
    return out


def main() -> None:
    mlflow.set_tracking_uri(str(MLFLOW_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)
    metrics = load_metrics()

    with mlflow.start_run(run_name="imported_models") as run:
        mlflow.log_param("classification_artifact", str(CLASS_PATH.name if CLASS_PATH.exists() else "missing"))
        mlflow.log_param("regression_artifact", str(REG_PATH.name if REG_PATH.exists() else "missing"))

        if "classification" in metrics:
            for k, v in metrics["classification"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"classification_{k}", float(v))
        if "regression" in metrics:
            for k, v in metrics["regression"].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"regression_{k}", float(v))

        # log model files as artifacts for reproducibility (no changes to files)
        if CLASS_PATH.exists():
            mlflow.log_artifact(str(CLASS_PATH), artifact_path="models")
        if REG_PATH.exists():
            mlflow.log_artifact(str(REG_PATH), artifact_path="models")

    print(f"Logged run to MLflow at {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
