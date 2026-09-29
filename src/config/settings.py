"""Project-level settings and path configuration."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

@dataclass(frozen=True)
class Settings:
    project_name: str = "airline-flight-delay-ml"
    python_version: str = "3.11"
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    interim_data_dir: Path = INTERIM_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR

SETTINGS = Settings()
