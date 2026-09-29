"""FastAPI application entry point."""

from fastapi import FastAPI, HTTPException

from api.routes import router
from api.schemas import FlightInput, PredictionResponse
from api.services import PredictionService

app = FastAPI(
    title="Airline Flight Delay Prediction API",
    description="Explainable AI-based airline delay prediction and decision support service.",
    version="0.1.0",
)

app.include_router(router)
service = PredictionService()


@app.get("/health")
def health() -> dict[str, str]:
    """Return the application health status."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict_flight(flight_input: FlightInput) -> PredictionResponse:
    """Generate a delay prediction for a valid pre-departure flight request."""
    try:
        return service.predict(flight_input)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
