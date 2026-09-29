"""Tests for package-level feature scaffolding."""

import pandas as pd

from src.features import __doc__ as feature_doc
from src.features.flight_features import add_flight_features


def test_feature_package_exists() -> None:
    """Feature package should remain available for future engineering modules."""
    assert isinstance(feature_doc, str)


def test_add_flight_features_creates_pre_departure_columns() -> None:
    """Flight features should produce a valid pre-departure modeling feature set."""
    df = pd.DataFrame(
        {
            "year": [2024],
            "month": [1],
            "day_of_month": [1],
            "day_of_week": [1],
            "fl_date": ["2024-01-01"],
            "op_unique_carrier": ["AA"],
            "origin": ["JFK"],
            "dest": ["LAX"],
            "crs_dep_time": [1200],
            "crs_arr_time": [1500],
            "crs_elapsed_time": [180.0],
            "distance": [2000.0],
        }
    )
    out = add_flight_features(df)
    assert "dep_hour" in out.columns
    assert "arr_hour" in out.columns
    assert "departure_period" in out.columns
    assert "route" in out.columns
    assert "is_weekend" in out.columns
