"""Data cleaning utilities for flight delay analysis."""

import pandas as pd

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with normalized column names."""
    df = df.copy()
    df.columns = [str(column).strip().lower().replace(" ", "_") for column in df.columns]
    return df
