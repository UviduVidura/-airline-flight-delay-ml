"""Simple monitoring checks: data quality and basic numeric distribution comparisons.

This script is conservative and uses the raw dataset as a reference baseline.
It does not claim production drift; it reports summary statistics and flags
large relative differences for numeric columns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.data_drift import detect_drift

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "flight_data_2024.csv"
OUT = ROOT / "reports" / "monitoring_summary.json"

# thresholds
NUMERIC_RELATIVE_THRESHOLD = 0.2  # 20% relative mean change flagged
TVD_THRESHOLD = 0.2


def summarize_numeric(df: pd.DataFrame) -> dict[str, Any]:
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    summary: dict[str, Any] = {"rows": len(df), "numeric": {}}
    for c in nums:
        col = df[c].dropna()
        if len(col) == 0:
            summary["numeric"][c] = {"count": 0}
            continue
        summary["numeric"][c] = {
            "count": int(col.count()),
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
        }
    return summary


def missing_value_monitor(df: pd.DataFrame) -> dict[str, Any]:
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    return {"missing_counts": missing, "total_rows": len(df)}


def category_check(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    cats = {}
    for c in current.select_dtypes(include=[object]).columns:
        ref_vals = set(reference[c].dropna().astype(str).unique())
        cur_vals = set(current[c].dropna().astype(str).unique())
        unseen = sorted(list(cur_vals - ref_vals))
        cats[c] = {"unseen_categories": unseen, "num_unseen": len(unseen)}
    return cats


def prediction_distribution_check(sample_df: pd.DataFrame) -> dict[str, Any]:
    """Run the saved model on a small sample and report prediction/probability distributions.

    This is an offline check and does not write to any external system.
    """
    try:
        from api.services import PredictionService
    except Exception:
        return {"error": "PredictionService unavailable"}

    svc = PredictionService()
    preds = []
    probs = []
    for _, row in sample_df.iterrows():
        try:
            payload = {k: (int(v) if pd.api.types.is_integer_dtype(type(v)) else v) for k, v in row.to_dict().items()}
            # Use FlightInput validation via service.predict
            from api.schemas import FlightInput

            fi = FlightInput(**payload)
            resp = svc.predict(fi)
            preds.append(int(resp.prediction))
            probs.append(float(resp.delay_probability))
        except Exception:
            continue

    if len(preds) == 0:
        return {"predictions": {"count": 0}}

    arr_probs = np.array(probs)
    return {
        "predictions": {"count": len(preds), "delay_rate": float(np.mean(preds))},
        "probability": {"mean": float(arr_probs.mean()), "std": float(arr_probs.std()), "min": float(arr_probs.min()), "max": float(arr_probs.max())},
    }


def main() -> None:
    if not RAW_PATH.exists():
        print("Reference raw file not found; skipping monitoring.")
        return

    df = pd.read_csv(RAW_PATH)
    # Use a reasonably large slice as reference and a smaller recent sample
    ref = df.head(50000).copy()
    cur = df.sample(n=min(1000, len(df)), random_state=42)

    summary_ref = summarize_numeric(ref)
    summary_cur = summarize_numeric(cur)

    missing_ref = missing_value_monitor(ref)
    missing_cur = missing_value_monitor(cur)

    category_issues = category_check(ref, cur)

    # drift detection
    drift_report = detect_drift(ref, cur, numeric_alpha=0.05, tvd_threshold=TVD_THRESHOLD)

    # run a small prediction distribution check using the model + pipeline
    sample_for_preds = cur.head(50).copy()
    # ensure feature engineering columns exist similar to training pipeline
    from src.data.cleaning import clean_flight_dataset
    from src.features.flight_features import add_flight_features

    sample_cleaned = clean_flight_dataset(sample_for_preds)
    sample_engineered = add_flight_features(sample_cleaned)
    sample_frame = sample_engineered.loc[:, [c for c in sample_engineered.columns if c in sample_engineered.columns]]

    prediction_stats = prediction_distribution_check(sample_engineered)

    out: dict[str, Any] = {
        "reference_summary": summary_ref,
        "current_summary": summary_cur,
        "missing": {"reference": missing_ref, "current": missing_cur},
        "categorical_issues": category_issues,
        "drift": drift_report,
        "prediction_stats": prediction_stats,
        "note": "Offline/Baseline Monitoring — does not imply production drift",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Monitoring summary written to {OUT}")


if __name__ == "__main__":
    main()
