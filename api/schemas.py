"""Pydantic schema definitions for API payloads."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlightInput(BaseModel):
    """Validated request model for a pre-departure flight delay prediction."""

    model_config = ConfigDict(extra="forbid")

    year: int = Field(..., ge=2023, le=2030, description="Flight year")
    month: int = Field(..., ge=1, le=12, description="Flight month")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month")
    day_of_week: int = Field(..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)")
    fl_date: date = Field(..., description="Local scheduled flight date")
    op_unique_carrier: str = Field(..., min_length=2, max_length=3, description="Airline carrier code")
    origin: str = Field(..., min_length=3, max_length=3, description="Origin airport code")
    dest: str = Field(..., min_length=3, max_length=3, description="Destination airport code")
    crs_dep_time: int = Field(..., ge=0, le=2359, description="Scheduled departure time as HHMM")
    crs_arr_time: int = Field(..., ge=0, le=2359, description="Scheduled arrival time as HHMM")
    crs_elapsed_time: float = Field(..., gt=0, description="Scheduled elapsed time in minutes")
    distance: float = Field(..., gt=0, description="Distance in miles")

    @field_validator("op_unique_carrier", "origin", "dest")
    @classmethod
    def uppercase_code(cls, value: str) -> str:
        """Normalize airport and carrier codes to uppercase."""
        return value.upper()

    @field_validator("fl_date")
    @classmethod
    def validate_date_matches_calendar_fields(cls, value: date, info) -> date:
        """Keep the request aligned to the calendar columns used by feature engineering."""
        data = info.data
        if isinstance(data, dict):
            year = data.get("year")
            month = data.get("month")
            day = data.get("day_of_month")
            if year is not None and month is not None and day is not None:
                if value.year != year or value.month != month or value.day != day:
                    raise ValueError("fl_date must match the supplied year, month, and day_of_month values.")
        return value


class FeatureContribution(BaseModel):
    """A single SHAP feature contribution."""

    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    """Response schema for a prediction result."""

    prediction: int
    classification: str
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_delay_minutes: float = Field(..., ge=0.0)
    important_contributing_features: list[FeatureContribution]
