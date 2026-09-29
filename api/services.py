"""Business logic for serving model predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from api.schemas import FeatureContribution, FlightInput, PredictionResponse
from src.data.cleaning import clean_flight_dataset
from src.explainability.shap_explainer import explain_flight_instance
from src.features.flight_features import add_flight_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_MODEL_PATH = PROJECT_ROOT / "models" / "classification_model.joblib"
REGRESSION_MODEL_PATH = PROJECT_ROOT / "models" / "regression_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
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


class PredictionService:
    """Load the persisted training artifacts and generate predictions for a valid flight request."""

    def __init__(self) -> None:
        self.name = "flight-delay-prediction-service"
        self.classification_model = self._load_classification_model()
        self.regression_model = self._load_regression_model()
        self.metadata = self._load_metadata()

    def _load_classification_model(self):
        """Load the saved classification pipeline without mutating the raw CSV."""
        if not CLASSIFICATION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
        return joblib.load(CLASSIFICATION_MODEL_PATH)

    def _load_regression_model(self):
        """Load the saved regression pipeline without mutating the raw CSV."""
        if not REGRESSION_MODEL_PATH.exists():
            raise FileNotFoundError(f"Regression model not found at {REGRESSION_MODEL_PATH}")
        return joblib.load(REGRESSION_MODEL_PATH)

    def _load_metadata(self) -> dict:
        """Return the saved model metadata for reporting and validation."""
        if not MODEL_METADATA_PATH.exists():
            return {}
        with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _prepare_features(flight_input: FlightInput | dict) -> pd.DataFrame:
        """Convert a valid pre-departure request into the feature matrix used by training."""
        payload = flight_input.model_dump() if isinstance(flight_input, FlightInput) else dict(flight_input)
        frame = pd.DataFrame([payload])
        frame["fl_date"] = pd.to_datetime(frame["fl_date"], errors="coerce")

        if "year" not in frame.columns:
            frame["year"] = frame["fl_date"].dt.year
        if "month" not in frame.columns:
            frame["month"] = frame["fl_date"].dt.month
        if "day_of_month" not in frame.columns:
            frame["day_of_month"] = frame["fl_date"].dt.day
        if "day_of_week" not in frame.columns:
            frame["day_of_week"] = frame["fl_date"].dt.dayofweek + 1

        frame = clean_flight_dataset(frame)
        frame = add_flight_features(frame)
        frame["route"] = frame["origin"].astype(str) + "_" + frame["dest"].astype(str)
        return frame[FEATURE_COLUMNS].copy()

    def predict(self, flight_input: FlightInput) -> PredictionResponse:
        """Return a delay classification, probability, regression estimate, and SHAP feature impacts."""
        feature_frame = self._prepare_features(flight_input)

        classification_model = self.classification_model
        regression_model = self.regression_model

        prediction = int(classification_model.predict(feature_frame)[0])
        delay_probability = float(classification_model.predict_proba(feature_frame)[0, 1])
        predicted_delay_minutes = max(float(regression_model.predict(feature_frame)[0]), 0.0)

        explanation = explain_flight_instance(model=classification_model, flight_input=feature_frame.to_dict(orient="records")[0])
        features = [
            FeatureContribution(feature=item["feature"], contribution=float(item["contribution"]))
            for item in explanation["important_contributing_features"]
        ]

        return PredictionResponse(
            prediction=prediction,
            classification="Delayed" if prediction == 1 else "On Time",
            delay_probability=delay_probability,
            predicted_delay_minutes=round(predicted_delay_minutes, 2),
            important_contributing_features=features,
        )

    def get_status(self) -> str:
        """Return service availability status."""
        return "ready"

    def get_model_info(self) -> dict:
        """Return model metadata useful for clients and dashboards.

        Falls back to a minimal summary when metadata file is missing.
        """
        if self.metadata:
            return self.metadata

        info = {
            "classification_model": getattr(self.classification_model.named_steps.get("model"), "__class__", str(type(self.classification_model))) if self.classification_model is not None else None,
            "regression_model": getattr(self.regression_model.named_steps.get("model"), "__class__", str(type(self.regression_model))) if self.regression_model is not None else None,
            "feature_columns": FEATURE_COLUMNS,
        }
        return info
