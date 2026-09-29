"""Preprocessing utilities for model training."""

import pandas as pd

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature preparation steps before training."""
    return df.copy()
