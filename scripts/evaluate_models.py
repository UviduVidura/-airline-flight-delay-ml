"""Evaluate the saved flight-delay models and verify the artifacts load correctly."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.data.cleaning import clean_flight_dataset
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.regression_metrics import compute_regression_metrics
from src.features.flight_features import add_flight_features

RAW_DATA_PATH = Path("data/raw/flight_data_2024.csv")
MODELS_DIR = Path("models")
FEATURE_COLUMNS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
    "dep_hour",
    "arr_hour",
    "departure_period",
    "season",
    "is_weekend",
    "route",
]


def load_eval_frame() -> pd.DataFrame:
    """Prepare the final evaluation dataset using the same flow as training."""
    frame = pd.read_csv(RAW_DATA_PATH)
    frame = clean_flight_dataset(frame)
    frame = add_flight_features(frame).sort_values("fl_date").reset_index(drop=True)
    return frame


def main() -> None:
    """Load the saved models and verify that they work on the holdout set."""
    eval_df = load_eval_frame()
    split_index = int(len(eval_df) * 0.8)
    X_test = eval_df.iloc[split_index:][FEATURE_COLUMNS]
    y_cls = eval_df.iloc[split_index:]["dep_del15"].astype(int)
    y_reg = eval_df.iloc[split_index:]["dep_delay"].astype(float)

    classification_model = joblib.load(MODELS_DIR / "classification_model.joblib")
    regression_model = joblib.load(MODELS_DIR / "regression_model.joblib")

    classification_proba = classification_model.predict_proba(X_test)[:, 1]
    classification_pred = (classification_proba >= 0.5).astype(int)
    regression_pred = regression_model.predict(X_test)

    classification_metrics = compute_classification_metrics(y_cls, classification_pred, y_proba=classification_proba, threshold=0.5)
    regression_metrics = compute_regression_metrics(y_reg, regression_pred)

    results = {
        "classification": classification_metrics,
        "regression": regression_metrics,
    }
    output_path = Path("reports/model_results/evaluation_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
