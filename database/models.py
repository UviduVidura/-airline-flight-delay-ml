"""ORM models for predictions and model metadata."""

from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from database.database import Base


class Prediction(Base):
    """Represents a stored prediction record."""

    __tablename__ = "predictions"

    id = Integer(primary_key=True, index=True)
    flight_number = String(50, nullable=False)
    origin = String(10, nullable=False)
    destination = String(10, nullable=False)
    delay_probability = Float(nullable=False)
    predicted_delay = Boolean(nullable=False)
    created_at = DateTime(timezone=True, server_default=func.now(), nullable=False)


class ModelVersion(Base):
    """Stores model version metadata."""

    __tablename__ = "model_versions"

    id = Integer(primary_key=True, index=True)
    model_name = String(150, nullable=False)
    version = String(50, nullable=False)
    framework = String(100, nullable=False)
    created_at = DateTime(timezone=True, server_default=func.now(), nullable=False)
    notes = String(500, nullable=True)
