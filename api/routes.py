"""API route definitions."""

from fastapi import APIRouter, HTTPException

from api.schemas import FlightInput, PredictionResponse
from api.services import PredictionService

router = APIRouter(prefix="/api", tags=["api"])
service = PredictionService()


@router.get("/status")
def route_status() -> dict[str, str]:
    """Return a basic service status payload."""
    return {"status": "ok", "service": "airline-flight-delay-ml"}


@router.post("/predict", response_model=PredictionResponse)
def route_predict(flight_input: FlightInput) -> PredictionResponse:
    """Generate a delay prediction for a valid pre-departure flight request."""
    try:
        return service.predict(flight_input)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/model-info")
def route_model_info() -> dict:
    """Return metadata about the loaded models and configuration."""
    try:
        return service.get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
