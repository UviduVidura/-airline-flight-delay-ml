"""Pydantic schema definitions for API payloads."""

from pydantic import BaseModel, Field

class FlightInput(BaseModel):
    """Example request model for flight delay predictions."""
    flight_number: str = Field(..., description="Flight number")
    origin: str = Field(..., description="Origin airport code")
    destination: str = Field(..., description="Destination airport code")
    scheduled_departure: str = Field(..., description="Scheduled departure time")

class PredictionResponse(BaseModel):
    """Response schema for a prediction result."""
    delay_probability: float
    classification: str
