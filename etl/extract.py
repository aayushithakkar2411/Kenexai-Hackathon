"""EXTRACT stage for the E-commerce churn ETL pipeline.

This module is intentionally designed as a reusable pipeline component:
- It loads the raw source data from Excel.
- It performs schema and column validation.
- It inspects data types for downstream transform/load compatibility.
- It returns a pandas DataFrame for the next ETL stage.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"

# IMPORTANT: specify the correct sheet name containing dataset
SOURCE_SHEET_NAME = "E Comm"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "CustomerID",
    "Churn",
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
)

OPTIONAL_DTYPE_EXPECTATIONS: dict[str, str] = {
    "CustomerID": "numeric",
    "Churn": "numeric",
    "Tenure": "numeric",
    "CityTier": "numeric",
    "SatisfactionScore": "numeric",
    "CashbackAmount": "numeric",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the module logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("etl.extract")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_file_exists(file_path: Path) -> None:
    """Ensure the source file exists before attempting to load."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file not found at '{file_path}'. "
            "Place the dataset at data/raw/E Commerce Dataset.xlsx"
        )


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Validate that all expected columns are present in the source data."""
    required_set = set(required_columns)
    existing_set = set(df.columns)
    missing = sorted(required_set - existing_set)

    if missing:
        raise ValueError(
            "Schema validation failed. Missing required columns: "
            + ", ".join(missing)
        )


def validate_schema_quality(df: pd.DataFrame) -> None:
    """Run generic schema-quality checks useful for production pipelines."""
    if df.empty:
        raise ValueError("Loaded dataset is empty. Extraction cannot continue.")

    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            "Schema validation failed. Duplicate columns detected: "
            + ", ".join(duplicated)
        )


def inspect_datatypes(df: pd.DataFrame, expected: dict[str, str]) -> None:
    """Inspect and log dtypes; warn if key fields are not in expected classes."""
    logger.info("Column datatype inspection:")

    for col, dtype in df.dtypes.items():
        logger.info("  - %s: %s", col, dtype)

    for col, expectation in expected.items():
        if col not in df.columns:
            continue

        if expectation == "numeric" and not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(
                "Column '%s' expected numeric type but found '%s'.",
                col,
                df[col].dtype,
            )


# ---------------------------------------------------------------------------
# Extract logic
# ---------------------------------------------------------------------------
def load_dataset(file_path: Path, sheet_name: str = SOURCE_SHEET_NAME) -> pd.DataFrame:
    """Load a CSV or Excel dataset into a DataFrame with robust error handling."""
    validate_file_exists(file_path)

    try:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            logger.info("Reading CSV source: %s", file_path)
            df = pd.read_csv(file_path)
        elif suffix in {".xlsx", ".xls"}:
            logger.info("Reading Excel sheet: %s", sheet_name)
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            raise ValueError(
                f"Unsupported source format '{suffix}'. Supported: .csv, .xlsx, .xls"
            )
    except ValueError as exc:
        raise ValueError(
            f"Failed to parse source file '{file_path}'. Check file integrity and sheet name."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error while reading '{file_path}': {exc}"
        ) from exc

    return df


def extract_data(file_path: Path | None = None, sheet_name: str = SOURCE_SHEET_NAME) -> pd.DataFrame:
    """Main extract function for ETL orchestration.

    Returns:
        pd.DataFrame: Validated raw input dataset.
    """
    if file_path is None:
        env_path = os.getenv("BATCH_INPUT_PATH")
        file_path = Path(env_path) if env_path else RAW_DATA_PATH

    logger.info("Starting EXTRACT stage.")
    logger.info("Source path: %s", file_path)

    df = load_dataset(file_path, sheet_name=sheet_name)

    validate_schema_quality(df)
    validate_required_columns(df, REQUIRED_COLUMNS)

    logger.info("Dataset loaded successfully with shape: rows=%d, cols=%d", *df.shape)

    inspect_datatypes(df, OPTIONAL_DTYPE_EXPECTATIONS)

    sample_size = min(5, len(df))
    logger.info("Sample preview (first %d rows):\n%s", sample_size, df.head(sample_size))

    logger.info("EXTRACT stage completed successfully.")

    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        extracted_df = extract_data()

        logger.info(
            "Returned DataFrame with %d rows for downstream stages.",
            len(extracted_df),
        )

    except FileNotFoundError as exc:
        logger.error("File error during extraction: %s", exc)
        raise

    except ValueError as exc:
        logger.error("Validation error during extraction: %s", exc)
        raise

    except Exception as exc:
        logger.exception("Fatal error in EXTRACT stage: %s", exc)
        raise
