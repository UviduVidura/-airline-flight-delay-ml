"""Streamlit dashboard for the airline flight delay prediction API."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"

def load_model_metadata() -> dict:
    """Load persisted model metadata for display in the dashboard."""
    if not MODEL_METADATA_PATH.exists():
        return {}
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_prediction_payload(
    flight_date: date | str,
    carrier: str,
    origin: str,
    destination: str,
    dep_time: str | int,
    arr_time: str | int,
    elapsed_time: float,
    distance: float,
) -> dict:
    """Build the payload expected by the FastAPI prediction endpoint."""
    parsed_date = date.fromisoformat(str(flight_date)) if isinstance(flight_date, str) else flight_date

    def _to_minutes(value: str | int) -> int:
        if isinstance(value, int):
            return value
        cleaned = str(value).strip().replace(":", "")
        if not cleaned:
            raise ValueError("Flight time must include a valid HH:MM or HHMM value.")
        return int(cleaned)

    dep_minutes = _to_minutes(dep_time)
    arr_minutes = _to_minutes(arr_time)
    return {
        "year": parsed_date.year,
        "month": parsed_date.month,
        "day_of_month": parsed_date.day,
        "day_of_week": parsed_date.isoweekday(),
        "fl_date": parsed_date.isoformat(),
        "op_unique_carrier": carrier.upper().strip(),
        "origin": origin.upper().strip(),
        "dest": destination.upper().strip(),
        "crs_dep_time": dep_minutes,
        "crs_arr_time": arr_minutes,
        "crs_elapsed_time": float(elapsed_time),
        "distance": float(distance),
    }


def get_backend_health(base_url: str) -> dict | None:
    """Check FastAPI status and return the JSON response when available."""
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def predict_from_backend(base_url: str, payload: dict) -> dict | None:
    """Send a prediction request to the API and return the JSON response."""
    try:
        response = requests.post(f"{base_url.rstrip('/')}/predict", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


st.set_page_config(page_title="Flight Delay Dashboard", page_icon="✈️", layout="wide")

st.title("Airline Flight Delay Prediction")
st.caption("Explainable AI decision support for pre-departure airline delays")

metadata = load_model_metadata()
backend_url = st.sidebar.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")
health = get_backend_health(backend_url)

if health is None:
    st.warning("The FastAPI backend is not reachable. Start it with: uvicorn api.main:app --host 127.0.0.1 --port 8000")
else:
    st.success(f"Backend status: {health.get('status', 'healthy')}")

st.markdown("### Flight Information")
with st.form("flight_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        flight_date = st.date_input("Scheduled flight date", value=pd.Timestamp("2024-01-01").date())
        carrier = st.text_input("Airline / carrier", value="9E")
        origin = st.text_input("Origin airport", value="JFK")
        destination = st.text_input("Destination airport", value="DTW")
    with col2:
        departure_time = st.text_input("Scheduled departure time (HH:MM)", value="12:52")
        arrival_time = st.text_input("Scheduled arrival time (HH:MM)", value="15:08")
        elapsed_time = st.number_input("Scheduled elapsed time (minutes)", min_value=30.0, max_value=600.0, value=136.0)
        distance = st.number_input("Distance (miles)", min_value=1.0, max_value=5000.0, value=509.0)

    submitted = st.form_submit_button("Predict delay")

if submitted:
    payload = build_prediction_payload(
        flight_date=flight_date,
        carrier=carrier,
        origin=origin,
        destination=destination,
        dep_time=departure_time,
        arr_time=arrival_time,
        elapsed_time=elapsed_time,
        distance=distance,
    )
    result = predict_from_backend(backend_url, payload)
    if result is None:
        st.error("Prediction request failed. Validate the backend URL or the request payload.")
    else:
        prediction = int(result.get("prediction", 0))
        probability = float(result.get("delay_probability", 0.0))
        delay_minutes = float(result.get("predicted_delay_minutes", 0.0))
        classification = result.get("classification", "Unknown")
        st.markdown("### Prediction result")
        if prediction == 1:
            st.error(f"Delayed: {classification}")
        else:
            st.success(f"On time: {classification}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", classification)
        col2.metric("Delay probability", f"{probability:.2%}")
        col3.metric("Predicted delay", f"{delay_minutes:.1f} min")

        explanation = result.get("important_contributing_features", [])
        if explanation:
            df = pd.DataFrame(explanation)
            df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")
            st.markdown("### Top contributing features")
            st.bar_chart(df.set_index("feature")["contribution"].sort_values(ascending=False).head(10))
            st.dataframe(df.head(10), use_container_width=True)
        else:
            st.info("No SHAP contributions were returned for this request.")

st.markdown("### Model information")
if metadata:
    st.write(f"Classification model: {metadata.get('classification_model', 'Unknown')}")
    st.write(f"Regression model: {metadata.get('regression_model', 'Unknown')}")
    st.write(f"Classification threshold: {metadata.get('classification_threshold', 'Unknown')}")
    st.write(f"Classification target: {metadata.get('classification_target', 'Unknown')}")
    st.write(f"Generated at: {metadata.get('generated_at', 'Unknown')}")
else:
    st.info("Model metadata is not available locally.")

st.markdown("### Project overview")
st.info("This dashboard calls the FastAPI backend and reuses the saved model artifacts, SHAP logic, and preprocessing pipeline from the project.")
