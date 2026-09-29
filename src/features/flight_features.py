"""Flight-specific feature engineering utilities."""

from __future__ import annotations

import pandas as pd


def _get_departure_period(minutes: int | float) -> str:
    """Map a scheduled departure time to a coarse period-of-day label."""
    if pd.isna(minutes):
        return "Unknown"
    hour = int(minutes // 100)
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def add_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful pre-departure flight features from the real schedule data."""
    features = df.copy()

    if "crs_dep_time" in features.columns:
        features["dep_hour"] = (features["crs_dep_time"] // 100).astype(int)
        features["departure_period"] = features["crs_dep_time"].apply(_get_departure_period)
    else:
        features["dep_hour"] = pd.NA
        features["departure_period"] = "Unknown"

    if "crs_arr_time" in features.columns:
        features["arr_hour"] = (features["crs_arr_time"] // 100).astype(int)
    else:
        features["arr_hour"] = pd.NA

    if "month" in features.columns:
        month_to_season = {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
        features["season"] = features["month"].map(month_to_season)
    else:
        features["season"] = "Unknown"

    if "day_of_week" in features.columns:
        features["is_weekend"] = features["day_of_week"].isin([6, 7]).astype(int)
    else:
        features["is_weekend"] = 0

    if {"origin", "dest"}.issubset(features.columns):
        features["route"] = features["origin"].astype(str) + "_" + features["dest"].astype(str)
    else:
        features["route"] = "Unknown"

    if "fl_date" in features.columns:
        features["fl_date"] = pd.to_datetime(features["fl_date"], errors="coerce")

    return features.reset_index(drop=True)
