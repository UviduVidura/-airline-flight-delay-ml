"""Tests for SHAP explanation generation."""

from src.explainability.shap_explainer import explain_flight_instance, load_saved_model


def test_explain_flight_instance_uses_saved_model() -> None:
    """The explanation helper should return a valid probability and feature contribution map."""
    model = load_saved_model()
    explanation = explain_flight_instance(
        model=model,
        flight_input={
            "year": 2024,
            "month": 1,
            "day_of_month": 1,
            "day_of_week": 1,
            "op_unique_carrier": "9E",
            "origin": "JFK",
            "dest": "DTW",
            "crs_dep_time": 1252,
            "crs_arr_time": 1508,
            "crs_elapsed_time": 136.0,
            "distance": 509.0,
            "dep_hour": 12,
            "arr_hour": 15,
            "departure_period": "Afternoon",
            "season": "Winter",
            "is_weekend": 0,
            "route": "JFK_DTW",
        },
    )

    assert "prediction" in explanation
    assert "delay_probability" in explanation
    assert "important_contributing_features" in explanation
    assert "shap_contribution_values" in explanation
    assert 0.0 <= explanation["delay_probability"] <= 1.0
    assert explanation["prediction"] in {0, 1}
