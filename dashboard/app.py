"""Streamlit landing page for the airline flight delay dashboard."""

import streamlit as st

st.set_page_config(page_title="Flight Delay Dashboard", page_icon="✈️", layout="wide")

st.title("Airline Flight Delay Prediction")
st.subheader("Explainable AI-Based Airline Flight Delay Prediction and Intelligent Decision Support System")

st.write(
    "A modular data science and MLOps project for airline delay prediction, explainability, and operational decision support."
)

status_col, stack_col = st.columns(2)

with status_col:
    st.markdown("### Project status")
    st.success("Setup and architecture complete")
    st.caption("Current stage: data ingestion, validation, and feature pipeline setup")

with stack_col:
    st.markdown("### Technology stack")
    st.code("Python 3.11\nFastAPI\nStreamlit\nscikit-learn\nXGBoost\nSHAP\nMLflow\nSQLAlchemy")

st.info("This landing page is intentionally set up for future prediction, analytics, and explainability views.")
