"""Tests for project data utilities."""

import pandas as pd

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
