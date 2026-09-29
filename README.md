# Airline Flight Delay Prediction and Decision Support System

This project predicts whether a flight is likely to be delayed using the saved sklearn pipeline artifacts and a SHAP-based explainability layer. It includes a FastAPI backend for operational prediction and a Streamlit dashboard for interactive use.

## Project architecture

- `api/`: FastAPI application, request schemas, prediction service, and routes.
- `dashboard/`: Streamlit dashboard that calls the FastAPI backend.
- `src/`: Core feature engineering, cleaning, evaluation, training, and explainability logic.
- `src/explainability/shap_explainer.py`: Reusable SHAP explanation helper used by both the API and the dashboard.
- `data/raw/`: Local raw flight CSV (not modified by the API or training workflow).
- `models/`: Saved serialized models and metadata generated from the completed training pipeline.
- `reports/`: Generated model evaluation outputs and SHAP figure files.
- `tests/`: Automated regression tests for model loading, data inputs, API behavior, and dashboard payload construction.

## How to start the project

```powershell
cd D:\airline-flight-delay-ml
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start the FastAPI backend

```powershell
cd D:\airline-flight-delay-ml
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Available endpoints:

- `GET /health` — health check
- `POST /predict` — pre-departure flight prediction
- `GET /api/status` — service status

- `GET /api/model-info` — model metadata and feature columns

Example `POST /predict` payload:

```json
{
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
  "distance": 509.0
}
```

Response fields:

- `prediction`: integer (0 = on time, 1 = delayed >= 15min)
- `classification`: human-friendly label
- `delay_probability`: probability in [0,1]
- `predicted_delay_minutes`: regression estimate (minutes)
- `important_contributing_features`: SHAP feature contributions

## Start the Streamlit dashboard

```powershell
cd D:\airline-flight-delay-ml
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py --server.headless true --server.port 8501
```

Then open the local Streamlit URL in a browser.

## API request example

Example `POST /predict` body:

```json
{
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
  "distance": 509.0
}
```

Response shape:

```json
{
  "prediction": 0,
  "classification": "On Time",
  "delay_probability": 0.07,
  "predicted_delay_minutes": 12.3,
  "important_contributing_features": [
    {"feature": "day_of_month", "contribution": -0.1446},
    {"feature": "origin", "contribution": -0.0181}
  ]
}
```

## Data leakage constraints

The API accepts only pre-departure information. It does not accept or use leakage columns such as:

- `dep_delay`, `dep_del15`
- `arr_delay`, `arr_time`, `dep_time`
- taxi and wheels-off/wheels-on columns
- delay-cause fields such as `carrier_delay`, `weather_delay`, `nas_delay`, `security_delay`, `late_aircraft_delay`
- cancelled/diverted flags

The feature engineering logic reuses the same pre-departure column set as the saved training pipeline.

## SHAP explainability

The project reuses the existing SHAP implementation in `src/explainability/shap_explainer.py` without replacing it. The API returns the top influential features for the selected flight, and the Streamlit dashboard displays the same values in a user-friendly chart and table.

## Model outputs

The saved classification and regression models are reused directly from the project artifacts:

- `models/classification_model.joblib`
- `models/regression_model.joblib`
- `models/model_metadata.json`

The model metadata includes the selected feature columns, model names, threshold, targets, metrics, and generated timestamp.

## Verification status

Current verified test run (local environment):

```
.\.venv\Scripts\python.exe -m pytest -q

16 passed, 5 warnings in ~39s
```

Monitoring and MLflow:

- A lightweight offline/baseline monitoring script was added and run; it writes `reports/monitoring_summary.json` using a KS-based numeric drift detector and categorical checks.
- Existing model metrics were imported into a local MLflow file store (`mlruns/`) using `scripts/log_mlflow.py` (requires `MLFLOW_ALLOW_FILE_STORE=true` when using file-backed tracking on some setups).

## Notes

- The raw dataset is preserved and not modified.
- The API and dashboard do not retrain any models.
- The project reuses the existing saved model artifacts and SHAP implementation.
