"""Reusable validation helpers for project datasets."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> list[str]:
    """Return a list of missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    return missing


def check_duplicate_rows(df: pd.DataFrame) -> int:
    """Return the number of duplicate rows in the dataset."""
    return int(df.duplicated().sum())


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the missing-value counts per column."""
    return df.isna().sum()


def check_invalid_values(df: pd.DataFrame, column: str, valid_values: Iterable[Any] | None = None) -> list[Any]:
    """Return invalid values in a column if a valid-value set is supplied."""
    series = df[column]
    if valid_values is None:
        return []
    valid_set = set(valid_values)
    return series[~series.isin(valid_set)].tolist()


def check_data_types(df: pd.DataFrame, expected_types: dict[str, Any]) -> dict[str, bool]:
    """Check whether columns match expected pandas dtypes."""
    results: dict[str, bool] = {}
    for column, expected_dtype in expected_types.items():
        if column not in df.columns:
            results[column] = False
            continue
        actual_dtype = df[column].dtype
        results[column] = str(actual_dtype) == str(expected_dtype)
    return results


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
    expected_types: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a basic validation pass and return summary information."""
    missing_columns = validate_required_columns(df, required_columns or [])
    duplicate_count = check_duplicate_rows(df)
    missing_values = check_missing_values(df)
    type_checks = check_data_types(df, expected_types or {})

    return {
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_count,
        "missing_values": missing_values.to_dict(),
        "data_type_checks": type_checks,
    }
