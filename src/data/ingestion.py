"""Data ingestion helpers for raw flight datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def discover_csv_files(raw_dir: str | Path) -> list[Path]:
    """Return CSV files found in a raw data directory."""
    path = Path(raw_dir)
    if not path.exists():
        return []
    return sorted(path.glob("*.csv"))


def read_csv_file(file_path: str | Path) -> pd.DataFrame:
    """Read a single CSV file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def check_columns(df: pd.DataFrame, required_columns: Iterable[str] | None = None) -> list[str]:
    """Return missing required columns, if any."""
    if required_columns is None:
        return []
    required = list(required_columns)
    return [column for column in required if column not in df.columns]


def report_row_counts(df: pd.DataFrame) -> int:
    """Return the number of rows in a DataFrame."""
    return len(df.index)


def combine_monthly_datasets(csv_files: Iterable[str | Path]) -> pd.DataFrame:
    """Combine multiple CSV files into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        frame = read_csv_file(csv_file)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def save_combined_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a combined dataset to disk."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def main() -> None:
    """Inspect available raw CSV files and print a summary."""
    raw_dir = Path("data/raw")
    csv_files = discover_csv_files(raw_dir)

    if not csv_files:
        print(f"No CSV files found in {raw_dir}. Add BTS data to this directory when available.")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {raw_dir}.")
    for file_path in csv_files:
        df = read_csv_file(file_path)
        print(f"- {file_path.name}: {len(df)} rows, {len(df.columns)} columns")

    combined = combine_monthly_datasets(csv_files)
    output_path = Path("data/interim/combined_data.csv")
    save_combined_dataset(combined, output_path)
    print(f"Saved combined dataset to {output_path}")


if __name__ == "__main__":
    main()
