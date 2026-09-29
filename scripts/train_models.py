"""Train the flight-delay classification and regression models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.data.cleaning import clean_flight_dataset
from src.evaluation.classification_metrics import compute_classification_metrics
from src.evaluation.regression_metrics import compute_regression_metrics
from src.features.flight_features import add_flight_features
from src.models.train_classifier import train_classifier_model
from src.models.train_regressor import train_regressor_model

RAW_DATA_PATH = Path("data/raw/flight_data_2024.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports/model_results")

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

LEAKAGE_COLUMNS = {
    "dep_delay",
    "dep_del15",
    "dep_time",
    "arr_delay",
    "arr_time",
    "wheels_off",
    "wheels_on",
    "taxi_out",
    "taxi_in",
    "actual_elapsed_time",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
    "cancelled",
    "cancellation_code",
    "diverted",
}


def build_model_metadata(results: dict, feature_columns: list[str], classification_threshold: float) -> dict:
    """Build the persisted metadata block for the trained models."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "classification_target": "dep_del15",
        "regression_target": "dep_delay",
        "classification_threshold": classification_threshold,
        "classification_model": results["classification"]["model_name"],
        "regression_model": results["regression"]["model_name"],
        "classification_path": "models/classification_model.joblib",
        "regression_path": "models/regression_model.joblib",
        "classification_metrics": results["classification"]["metrics"],
        "regression_metrics": results["regression"]["metrics"],
        "leakage_columns_excluded": sorted(LEAKAGE_COLUMNS),
    }


def load_dataset() -> pd.DataFrame:
    """Load the real raw dataset and apply the project cleaning and feature engineering steps."""
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned = clean_flight_dataset(raw_df)
    engineered = add_flight_features(cleaned)
    return engineered.sort_values("fl_date").reset_index(drop=True)


def train_classification_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, object]:
    """Train and score the allowed classification candidates."""
    model_candidates = [
        ("DummyClassifier", DummyClassifier(strategy="stratified")),
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=42)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ]

    try:
        import xgboost as xgb  # type: ignore

        model_candidates.append(
            ("XGBoostClassifier", xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            ))
        )
    except Exception as exc:  # pragma: no cover - runtime dependency fallback
        print(f"XGBoostClassifier unavailable: {type(exc).__name__}: {exc}")

    best_result = None
    best_model = None
    for model_name, model in model_candidates:
        fitted_model = train_classifier_model(X_train, y_train, model=model)
        y_proba = fitted_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = compute_classification_metrics(y_test, y_pred, y_proba=y_proba, threshold=0.5)
        result = {"model_name": model_name, "metrics": metrics}
        if best_result is None or metrics["f1"] > best_result["metrics"]["f1"]:
            best_result = result
            best_model = fitted_model

    return best_result, best_model


def train_regression_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, object]:
    """Train and score the allowed regression candidates."""
    model_candidates = [
        ("DummyRegressor", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ]

    try:
        import xgboost as xgb  # type: ignore

        model_candidates.append(
            ("XGBoostRegressor", xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ))
        )
    except Exception as exc:  # pragma: no cover - runtime dependency fallback
        print(f"XGBoostRegressor unavailable: {type(exc).__name__}: {exc}")

    best_result = None
    best_model = None
    for model_name, model in model_candidates:
        fitted_model = train_regressor_model(X_train, y_train, model=model)
        y_pred = fitted_model.predict(X_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        result = {"model_name": model_name, "metrics": metrics}
        if best_result is None or metrics["r2"] > best_result["metrics"]["r2"]:
            best_result = result
            best_model = fitted_model

    return best_result, best_model


def main() -> None:
    """Run the full classification and regression training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    feature_frame = df[FEATURE_COLUMNS].copy()

    classification_target = df["dep_del15"].astype(int)
    regression_target = df["dep_delay"].astype(float)

    split_index = int(len(df) * 0.8)
    X_train_cls = feature_frame.iloc[:split_index].copy()
    X_test_cls = feature_frame.iloc[split_index:].copy()
    y_train_cls = classification_target.iloc[:split_index].copy()
    y_test_cls = classification_target.iloc[split_index:].copy()

    X_train_reg = feature_frame.iloc[:split_index].copy()
    X_test_reg = feature_frame.iloc[split_index:].copy()
    y_train_reg = regression_target.iloc[:split_index].copy()
    y_test_reg = regression_target.iloc[split_index:].copy()

    classification_result, classification_model = train_classification_models(X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    regression_result, regression_model = train_regression_models(X_train_reg, y_train_reg, X_test_reg, y_test_reg)

    classification_metrics_path = REPORTS_DIR / "classification_metrics.json"
    regression_metrics_path = REPORTS_DIR / "regression_metrics.json"
    classification_metrics_path.write_text(json.dumps(classification_result["metrics"], indent=2), encoding="utf-8")
    regression_metrics_path.write_text(json.dumps(regression_result["metrics"], indent=2), encoding="utf-8")

    classification_model_path = MODELS_DIR / "classification_model.joblib"
    regression_model_path = MODELS_DIR / "regression_model.joblib"
    joblib.dump(classification_model, classification_model_path)
    joblib.dump(regression_model, regression_model_path)

    metadata = build_model_metadata(
        {
            "classification": classification_result,
            "regression": regression_result,
        },
        FEATURE_COLUMNS,
        classification_threshold=0.5,
    )
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Classification best model:", classification_result["model_name"])
    print(json.dumps(classification_result["metrics"], indent=2))
    print("Regression best model:", regression_result["model_name"])
    print(json.dumps(regression_result["metrics"], indent=2))
    print(f"Saved classification model: {classification_model_path}")
    print(f"Saved regression model: {regression_model_path}")
    print(f"Saved metadata: {MODELS_DIR / 'model_metadata.json'}")


if __name__ == "__main__":
    main()
