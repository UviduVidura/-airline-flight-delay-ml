"""Data cleaning utilities for flight delay analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with normalized column names."""
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    return df


def clean_flight_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw flight dataset while preserving the original CSV untouched.

    The cleaning logic is intentionally conservative: it removes duplicates,
    drops rows without a usable departure delay, and removes rows flagged as
    cancelled when the dataset contains a cancelled indicator. The result is a
    supervised training dataset for pre-departure delay prediction.
    """
    cleaned = df.copy()
    cleaned = standardize_column_names(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if "fl_date" in cleaned.columns:
        cleaned["fl_date"] = pd.to_datetime(cleaned["fl_date"], errors="coerce")

    if "cancelled" in cleaned.columns:
        cleaned = cleaned[cleaned["cancelled"].fillna(0).astype(float) == 0].copy()

    if "dep_delay" in cleaned.columns:
        cleaned = cleaned[cleaned["dep_delay"].notna()].copy()

    if "dep_del15" not in cleaned.columns and "dep_delay" in cleaned.columns:
        cleaned["dep_del15"] = (cleaned["dep_delay"] >= 15).astype(int)

    if "dep_del15" in cleaned.columns:
        cleaned = cleaned[cleaned["dep_del15"].notna()].copy()

    return cleaned.reset_index(drop=True)


def prepare_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare a supervised dataset with pre-departure features and targets."""
    dataset = clean_flight_dataset(df)
    target_columns = {"dep_delay", "dep_del15"}
    feature_columns = [
        column
        for column in dataset.columns
        if column not in {"dep_delay", "dep_del15", "dep_time", "arr_time", "wheels_off", "wheels_on",
                            "taxi_out", "taxi_in", "actual_elapsed_time", "carrier_delay",
                            "weather_delay", "nas_delay", "security_delay",
                            "late_aircraft_delay", "cancelled", "cancellation_code",
                            "diverted"}
    ]
    return dataset[sorted({*feature_columns, *target_columns})].copy()


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist the cleaned data under data/processed without changing the raw CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
