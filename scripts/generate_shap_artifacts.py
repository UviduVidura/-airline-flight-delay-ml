"""Generate SHAP summary and bar plots for the saved classification model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.cleaning import clean_flight_dataset
from src.explainability.shap_explainer import explain_flight_instance
from src.features.flight_features import add_flight_features

FEATURE_COLUMNS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "op_unique_carrier",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
    "dep_hour",
    "arr_hour",
    "departure_period",
    "season",
    "is_weekend",
    "route",
]


def main() -> None:
    """Generate SHAP figure files and print an example explanation."""
    model = joblib.load(ROOT / "models" / "classification_model.joblib")
    raw = pd.read_csv(ROOT / "data" / "raw" / "flight_data_2024.csv")
    df = add_flight_features(clean_flight_dataset(raw)).sort_values("fl_date").reset_index(drop=True)
    sample = df[FEATURE_COLUMNS].copy().sample(n=min(10, len(df)), random_state=42)

    preprocessor = model.named_steps["preprocessor"]
    base_model = model.named_steps["model"]
    transformed = preprocessor.transform(sample)
    feature_names = [
        name.split("__", 1)[1] if "__" in name else name
        for name in preprocessor.get_feature_names_out().tolist()
    ]

    explainer = shap.TreeExplainer(base_model)
    values = explainer.shap_values(transformed)
    values_array = np.asarray(values)
    if values_array.ndim == 3:
        if values_array.shape[-1] == 2:
            values_array = values_array[..., 1]
        elif values_array.shape[0] == 2:
            values_array = values_array[1]
    elif values_array.ndim == 2 and values_array.shape[0] == 2:
        values_array = values_array[1]

    output_dir = ROOT / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(values_array, transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight", dpi=200)
    plt.close()

    global_importance = np.abs(values_array).mean(axis=0)
    ordered_index = np.argsort(global_importance)[::-1][:10]
    ordered_features = [feature_names[i] for i in ordered_index]
    ordered_scores = [float(global_importance[i]) for i in ordered_index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered_features[::-1], ordered_scores[::-1])
    ax.set_title("Global SHAP Feature Importance")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    explanation = explain_flight_instance(model=model)
    print({"summary_path": str(output_dir / "shap_summary.png"), "bar_path": str(output_dir / "shap_bar.png")})
    print(explanation)


if __name__ == "__main__":
    main()
