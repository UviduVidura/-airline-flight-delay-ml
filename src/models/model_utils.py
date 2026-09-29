"""Utility functions for model persistence and metadata."""

from pathlib import Path
import joblib

def save_model(model, output_path: str | Path):
    """Persist a trained model artifact to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(model_path: str | Path):
    """Load a saved model artifact."""
    return joblib.load(model_path)
