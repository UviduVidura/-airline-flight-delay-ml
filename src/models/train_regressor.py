"""Regressor training utilities for delay duration prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def build_regressor_pipeline(X: pd.DataFrame, model=None) -> Pipeline:
    """Create a preprocessing + modeling pipeline for regression training."""
    frame = X.copy()
    categorical_columns = [
        column for column in frame.columns if pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])
    ]
    numeric_columns = [column for column in frame.columns if column not in categorical_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    if model is None:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_regressor_model(X, y, model=None):
    """Train a regressor for delay duration estimation."""
    working = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    pipeline = build_regressor_pipeline(working, model=model)
    pipeline.fit(working, y)
    return pipeline
