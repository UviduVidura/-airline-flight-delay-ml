"""Tests for project data utilities."""

import pandas as pd

from src.data.cleaning import clean_flight_dataset, prepare_training_dataset
from src.data.validation import validate_dataframe, validate_required_columns


def test_project_imports() -> None:
    """Ensure the package imports without errors."""
    import src
    assert src.__name__ == "src"


def test_validate_required_columns() -> None:
    """Required columns should be recognized correctly."""
    df = pd.DataFrame({"flight_number": ["AA123"], "origin": ["JFK"]})
    missing = validate_required_columns(df, ["flight_number", "origin", "destination"])
    assert missing == ["destination"]


def test_validate_dataframe_summary() -> None:
    """Validation summary should report missing columns and duplicates."""
    df = pd.DataFrame(
        {
            "flight_number": ["AA123", "AA123"],
            "origin": ["JFK", "JFK"],
            "destination": ["LAX", "LAX"],
        }
    )
    summary = validate_dataframe(df, required_columns=["flight_number", "origin", "destination"])
    assert summary["duplicate_rows"] == 1
    assert summary["missing_columns"] == []


def test_clean_flight_dataset_removes_duplicates_and_invalid_rows() -> None:
    """Cleaning should drop duplicate and unsupervised rows while keeping usable targets."""
    df = pd.DataFrame(
        {
            "year": [2024, 2024, 2024],
            "month": [1, 1, 1],
            "day_of_month": [1, 1, 2],
            "day_of_week": [1, 1, 2],
            "fl_date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "op_unique_carrier": ["AA", "AA", "DL"],
            "origin": ["JFK", "JFK", "LAX"],
            "dest": ["LAX", "LAX", "JFK"],
            "crs_dep_time": [1200, 1200, 1400],
            "crs_arr_time": [1500, 1500, 1700],
            "crs_elapsed_time": [180.0, 180.0, 180.0],
            "distance": [2000.0, 2000.0, 2200.0],
            "dep_delay": [10.0, 20.0, None],
            "dep_del15": [0, 1, 0],
        }
    )
    cleaned = clean_flight_dataset(df)
    assert len(cleaned) == 2
    assert "dep_del15" in cleaned.columns
    assert cleaned["dep_delay"].notna().all()


def test_prepare_training_dataset_uses_pre_departure_features() -> None:
    """Training data should contain only pre-departure feature columns and the delay targets."""
    df = pd.DataFrame(
        {
            "year": [2024, 2024],
            "month": [1, 1],
            "day_of_month": [1, 2],
            "day_of_week": [1, 2],
            "fl_date": ["2024-01-01", "2024-01-02"],
            "op_unique_carrier": ["AA", "DL"],
            "origin": ["JFK", "LAX"],
            "dest": ["LAX", "JFK"],
            "crs_dep_time": [1200, 1400],
            "crs_arr_time": [1500, 1700],
            "crs_elapsed_time": [180.0, 180.0],
            "distance": [2000.0, 2200.0],
            "dep_delay": [10.0, 30.0],
            "dep_del15": [0, 1],
            "dep_hour": [12, 14],
            "arr_hour": [15, 17],
            "departure_period": ["Afternoon", "Afternoon"],
            "season": ["Winter", "Winter"],
            "is_weekend": [0, 0],
            "route": ["JFK_LAX", "LAX_JFK"],
        }
    )
    prepared = prepare_training_dataset(df)
    assert {"dep_delay", "dep_del15"}.issubset(prepared.columns)
    assert "dep_hour" in prepared.columns
    assert "origin" in prepared.columns
