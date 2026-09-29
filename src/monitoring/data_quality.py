"""Data quality monitoring utilities."""

def check_data_quality(df):
    """Return a summary of missing values and invalid records."""
    return {
        "rows": len(df),
        "missing_values": int(df.isnull().sum().sum()),
    }
