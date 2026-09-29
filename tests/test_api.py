"""Tests for the FastAPI app and dashboard integration."""

from fastapi.testclient import TestClient

from api.main import app
from api.services import PredictionService
from dashboard import app as dashboard_app

client = TestClient(app)

VALID_PAYLOAD = {
    "year": 2024,
    "month": 1,
    "day_of_month": 1,
    "day_of_week": 1,
    "fl_date": "2024-01-01",
    "op_unique_carrier": "9E",
    "origin": "JFK",
    "dest": "DTW",
    "crs_dep_time": 1252,
    "crs_arr_time": 1508,
    "crs_elapsed_time": 136.0,
    "distance": 509.0,
}


def test_health_endpoint() -> None:
    """The health endpoint should respond with a healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint_accepts_valid_flight_input() -> None:
    """A valid pre-departure flight payload should return a prediction response with ranges and SHAP features."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "prediction" in payload
    assert "classification" in payload
    assert "delay_probability" in payload
    assert "predicted_delay_minutes" in payload
    assert "important_contributing_features" in payload
    assert payload["prediction"] in {0, 1}
    assert 0.0 <= float(payload["delay_probability"]) <= 1.0
    assert float(payload["predicted_delay_minutes"]) >= 0.0
    assert isinstance(payload["important_contributing_features"], list)
    assert len(payload["important_contributing_features"]) > 0


def test_predict_endpoint_rejects_invalid_input() -> None:
    """The API should reject invalid or leaky features, such as non-pre-departure delay columns."""
    bad_payload = {**VALID_PAYLOAD, "dep_delay": 15}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_prediction_service_loads_models() -> None:
    """The prediction service should load the persisted training artifacts without retraining."""
    service = PredictionService()
    assert service.classification_model is not None
    assert service.regression_model is not None
    assert isinstance(service.metadata, dict)


def test_dashboard_helpers_build_valid_payload() -> None:
    """The dashboard should build a backend-compatible payload using the same pre-departure fields."""
    payload = dashboard_app.build_prediction_payload(
        flight_date=VALID_PAYLOAD["fl_date"],
        carrier=VALID_PAYLOAD["op_unique_carrier"],
        origin=VALID_PAYLOAD["origin"],
        destination=VALID_PAYLOAD["dest"],
        dep_time="12:52",
        arr_time="15:08",
        elapsed_time=VALID_PAYLOAD["crs_elapsed_time"],
        distance=VALID_PAYLOAD["distance"],
    )
    assert payload["op_unique_carrier"] == "9E"
    assert payload["origin"] == "JFK"
    assert payload["dest"] == "DTW"
    assert payload["crs_dep_time"] == 1252
    assert payload["crs_arr_time"] == 1508
    assert payload["crs_elapsed_time"] == 136.0
    assert payload["distance"] == 509.0


def test_model_info_endpoint() -> None:
    """The model-info endpoint should return metadata about saved models or feature columns."""
    response = client.get("/api/model-info")
    assert response.status_code == 200, response.text
    payload = response.json()
    # Should include feature_columns or classification_model keys
    assert isinstance(payload, dict)
    assert ("feature_columns" in payload) or ("classification_model" in payload)
