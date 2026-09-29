"""ORM models for predictions and model metadata."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from database.database import Base


class Prediction(Base):
    """Represents a stored prediction record."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    flight_number = Column(String(50), nullable=True)
    origin = Column(String(10), nullable=False)
    destination = Column(String(10), nullable=False)
    delay_probability = Column(Float, nullable=False)
    predicted_delay = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelVersion(Base):
    """Stores model version metadata."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(150), nullable=False)
    version = Column(String(50), nullable=False)
    framework = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    notes = Column(String(500), nullable=True)
