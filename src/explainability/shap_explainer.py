"""SHAP-based explainability utilities for the saved classification pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.cleaning import clean_flight_dataset
from src.features.flight_features import add_flight_features

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "models" / "classification_model.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "flight_data_2024.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
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


def load_saved_model():
    """Load the saved classification model from disk."""
    return joblib.load(MODEL_PATH)


def load_model_metadata() -> dict:
    """Load the persisted model metadata for feature and threshold configuration."""
    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_dataframe(record: dict | pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Normalize a record or DataFrame into a feature-only DataFrame."""
    if isinstance(record, pd.DataFrame):
        frame = record.copy()
    elif isinstance(record, dict):
        frame = pd.DataFrame([record])
    elif isinstance(record, list):
        frame = pd.DataFrame(record)
    else:
        raise TypeError("Input must be a dict, list of dicts, or pandas DataFrame.")

    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required flight features: {missing}")

    return frame[FEATURE_COLUMNS].copy()


def _get_feature_names(model) -> list[str]:
    """Return the final transformed feature names used by the trained pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    return [name.split("__", 1)[1] if "__" in name else name for name in preprocessor.get_feature_names_out().tolist()]


def _sample_reference_data(model, n_samples: int = 10) -> np.ndarray:
    """Load a compact representative background sample that keeps SHAP stable for this saved forest model."""
    reference = clean_flight_dataset(pd.read_csv(RAW_DATA_PATH))
    reference = add_flight_features(reference).sort_values("fl_date").reset_index(drop=True)
    sample = reference[FEATURE_COLUMNS].copy().sample(n=min(n_samples, len(reference)), random_state=42)
    return model.named_steps["preprocessor"].transform(sample)


def _get_shap_values(explainer, transformed: np.ndarray) -> np.ndarray:
    """Normalize SHAP outputs across supported SHAP versions and binary classification models."""
    raw_values = explainer.shap_values(transformed)
    array = np.asarray(raw_values.values if hasattr(raw_values, "values") else raw_values)
    if array.ndim == 3:
        if array.shape[-1] == 2:
            array = array[..., 1]
        elif array.shape[0] == 2:
            array = array[1]
        else:
            array = array[0]
    elif array.ndim == 2 and array.shape[0] == 2:
        array = array[1]
    return np.asarray(array)


def explain_flight_instance(model=None, flight_input: dict | pd.DataFrame | list[dict] | None = None) -> dict:
    """Explain a single real pre-departure flight instance using the saved pipeline and SHAP."""
    if model is None:
        model = load_saved_model()
    if flight_input is None:
        flight_input = {
            "year": 2024,
            "month": 1,
            "day_of_month": 1,
            "day_of_week": 1,
            "op_unique_carrier": "9E",
            "origin": "JFK",
            "dest": "DTW",
            "crs_dep_time": 1252,
            "crs_arr_time": 1508,
            "crs_elapsed_time": 136.0,
            "distance": 509.0,
            "dep_hour": 12,
            "arr_hour": 15,
            "departure_period": "Afternoon",
            "season": "Winter",
            "is_weekend": 0,
            "route": "JFK_DTW",
        }

    frame = _to_dataframe(flight_input)
    transformed = model.named_steps["preprocessor"].transform(frame)
    prediction = int(model.predict(frame)[0])
    delay_probability = float(model.predict_proba(frame)[0, 1])

    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = _get_shap_values(explainer, transformed)
    contribution_vector = shap_values[0] if shap_values.ndim == 2 else shap_values
    feature_names = _get_feature_names(model)
    contribution_map = {
        feature_name: float(value)
        for feature_name, value in zip(feature_names, contribution_vector)
    }

    top_features = [
        {"feature": feature_name, "contribution": float(value)}
        for feature_name, value in sorted(
            contribution_map.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:10]
    ]

    return {
        "prediction": prediction,
        "delay_probability": delay_probability,
        "important_contributing_features": top_features,
        "shap_contribution_values": contribution_map,
    }


def generate_shap_visualizations(model=None, sample_size: int = 10) -> dict[str, Path]:
    """Generate the summary and bar plots used for global SHAP feature importance.

    The saved forest model triggers SHAP's tree backend memory issues when the sample grows beyond a
    small representative subset in SHAP 0.52; using a compact sample keeps the explanation stable while
    preserving the same preprocessing and feature mapping used during training.
    """
    if model is None:
        model = load_saved_model()

    sample_size = max(1, min(int(sample_size), 10))
    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned = clean_flight_dataset(raw_df)
    engineered = add_flight_features(cleaned).sort_values("fl_date").reset_index(drop=True)
    sample_frame = engineered[FEATURE_COLUMNS].copy().sample(n=min(sample_size, len(engineered)), random_state=42)

    transformed = model.named_steps["preprocessor"].transform(sample_frame)
    explainer = shap.TreeExplainer(model.named_steps["model"])
    shap_values = _get_shap_values(explainer, transformed)
    feature_names = _get_feature_names(model)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = FIGURES_DIR / "shap_summary.png"
    bar_path = FIGURES_DIR / "shap_bar.png"

    shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight", dpi=200)
    plt.close()

    global_importance = np.abs(shap_values).mean(axis=0)
    ordered_index = np.argsort(global_importance)[::-1]
    ordered_features = [feature_names[idx] for idx in ordered_index[:10]]
    ordered_scores = [float(global_importance[idx]) for idx in ordered_index[:10]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered_features[::-1], ordered_scores[::-1])
    ax.set_title("Global SHAP Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    return {"summary": summary_path, "bar": bar_path}


def main() -> None:
    """Generate the SHAP report artifacts and print an example explanation."""
    model = load_saved_model()
    outputs = generate_shap_visualizations(model=model)
    explanation = explain_flight_instance(model=model)
    print({"summary": str(outputs["summary"]), "bar": str(outputs["bar"])})
    print(explanation)


if __name__ == "__main__":
    main()
