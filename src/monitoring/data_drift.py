"""Data drift detection utilities.

Provides a KS-test based numeric detector and a categorical distribution
comparison with sensible fallbacks when samples are small or empty.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
except Exception:  # pragma: no cover - scipy should be in requirements but fall back safely
    ks_2samp = None  # type: ignore


def _safe_array(series: pd.Series) -> np.ndarray:
    """Return a 1-d numpy array with NaNs removed for testing."""
    return series.dropna().to_numpy()


def numeric_ks_test(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Run a two-sample KS test on numeric data and return stats.

    If `scipy.stats.ks_2samp` is unavailable, return a conservative result
    indicating no-drift but include sample sizes so callers can decide.
    """
    ref = _safe_array(reference)
    cur = _safe_array(current)
    out = {"ref_count": int(ref.size), "cur_count": int(cur.size)}
    if ref.size < 2 or cur.size < 2:
        out.update({"statistic": None, "pvalue": None, "drift": False, "reason": "insufficient_samples"})
        return out

    if ks_2samp is None:
        out.update({"statistic": None, "pvalue": None, "drift": False, "reason": "scipy_unavailable"})
        return out

    res = ks_2samp(ref, cur)
    out.update({"statistic": float(res.statistic), "pvalue": float(res.pvalue), "drift": False})
    return out


def categorical_compare(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Compare categorical distributions by reporting frequencies, unseen categories,
    and total variation distance (TVD)."""
    ref = reference.dropna().astype(str)
    cur = current.dropna().astype(str)
    ref_counts = ref.value_counts(normalize=True)
    cur_counts = cur.value_counts(normalize=True)

    all_keys = sorted(set(ref_counts.index).union(set(cur_counts.index)))
    ref_vec = np.array([ref_counts.get(k, 0.0) for k in all_keys], dtype=float)
    cur_vec = np.array([cur_counts.get(k, 0.0) for k in all_keys], dtype=float)

    tvd = 0.5 * np.abs(ref_vec - cur_vec).sum()
    unseen_in_ref = [k for k in cur_counts.index if k not in ref_counts.index]
    unseen_in_cur = [k for k in ref_counts.index if k not in cur_counts.index]

    return {
        "tvd": float(tvd),
        "unseen_in_ref": unseen_in_ref,
        "unseen_in_cur": unseen_in_cur,
        "ref_top": ref_counts.head(10).to_dict(),
        "cur_top": cur_counts.head(10).to_dict(),
        "ref_count": int(ref.size),
        "cur_count": int(cur.size),
    }


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_alpha: float = 0.05, tvd_threshold: float = 0.2) -> dict[str, Any]:
    """Compare reference and current dataframes and report per-column drift info.

    Returns a dict mapping column -> drift report. Numeric columns get KS-test output;
    categorical columns get TVD and unseen-category reports. The caller can decide
    how to interpret p-values / TVD against thresholds.
    """
    report: dict[str, Any] = {}
    if reference_df is None or current_df is None:
        return report

    # Align columns present in either frame
    shared = [c for c in reference_df.columns if c in current_df.columns]

    for col in shared:
        ref_col = reference_df[col]
        cur_col = current_df[col]
        if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
            ks = numeric_ks_test(ref_col, cur_col)
            if ks.get("pvalue") is not None:
                ks["drift"] = ks.get("pvalue", 1.0) < numeric_alpha
            report[col] = {"type": "numeric", **ks}
        else:
            cat = categorical_compare(ref_col, cur_col)
            cat["drift"] = cat.get("tvd", 0.0) > tvd_threshold
            report[col] = {"type": "categorical", **cat}

    return report

