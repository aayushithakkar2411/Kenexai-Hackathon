"""Data profiling script for the EXTRACT output dataset.

This module analyzes data quality and schema characteristics without modifying
source data. It relies on etl.extract.extract_data() as the single source for
loading the dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from etl.extract import extract_data


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_REPORT_PATH = PROJECT_ROOT / "data" / "data_profile_report.csv"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger for profiling execution."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("etl.profile_data")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------
def build_profile_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a column-level profiling report with core data quality metrics."""
    missing_values = df.isna().sum()
    missing_percentage = (missing_values / len(df) * 100).round(2)

    report_df = pd.DataFrame(
        {
            "column_name": df.columns,
            "data_type": [str(dtype) for dtype in df.dtypes.values],
            "missing_values": [int(missing_values[col]) for col in df.columns],
            "missing_percentage": [float(missing_percentage[col]) for col in df.columns],
            "unique_values": [int(df[col].nunique(dropna=True)) for col in df.columns],
        }
    )

    return report_df


def identify_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numerical and categorical columns automatically."""
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_cols, categorical_cols


def get_numerical_summary(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    """Generate basic statistical summary for numerical columns."""
    if not numerical_cols:
        return pd.DataFrame()
    return df[numerical_cols].describe().transpose()


def save_profile_report(report_df: pd.DataFrame, report_path: Path = PROFILE_REPORT_PATH) -> None:
    """Save profiling report to CSV for downstream quality review."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)
    logger.info("Profile report saved at: %s", report_path)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run_data_profiling() -> pd.DataFrame:
    """Run complete profiling workflow and return the profiling report DataFrame."""
    logger.info("Starting data profiling workflow.")

    df = extract_data()

    logger.info("Dataset shape: rows=%d, cols=%d", *df.shape)
    logger.info("Top 5 rows:\n%s", df.head())

    report_df = build_profile_report(df)
    numerical_cols, categorical_cols = identify_column_groups(df)
    numerical_summary = get_numerical_summary(df, numerical_cols)

    logger.info("Numerical columns (%d): %s", len(numerical_cols), numerical_cols)
    logger.info("Categorical columns (%d): %s", len(categorical_cols), categorical_cols)

    missing_only = report_df[report_df["missing_values"] > 0][
        ["column_name", "missing_values", "missing_percentage"]
    ]

    if missing_only.empty:
        logger.info("Columns with missing values: none")
    else:
        logger.info("Columns with missing values:\n%s", missing_only.to_string(index=False))

    if numerical_summary.empty:
        logger.info("Numerical summary: no numerical columns found.")
    else:
        logger.info("Basic statistical summary (numerical columns):\n%s", numerical_summary)

    save_profile_report(report_df)

    logger.info("Data profiling workflow completed successfully.")
    return report_df


if __name__ == "__main__":
    run_data_profiling()
