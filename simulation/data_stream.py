"""Simulate a real-time data source for Bronze layer ingestion.

This script reads the full churn dataset and emits it in small CSV batches
at fixed time intervals to mimic a streaming source.
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
SOURCE_SHEET_NAME = "E Comm"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
BRONZE_DB_PATH = BRONZE_DIR / "bronze_layer.db"
BRONZE_TABLE_NAME = "bronze_customer_churn_events"
DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY_SECONDS = 5

INFORMATIVE_COLUMN_MAP = {
    "CustomerID": "customer_id",
    "Churn": "churn_flag",
    "Tenure": "customer_tenure_months",
    "PreferredLoginDevice": "preferred_login_device",
    "CityTier": "city_tier",
    "WarehouseToHome": "warehouse_to_home_distance",
    "PreferredPaymentMode": "preferred_payment_mode",
    "Gender": "customer_gender",
    "HourSpendOnApp": "hours_spent_on_app",
    "NumberOfDeviceRegistered": "registered_device_count",
    "PreferedOrderCat": "preferred_order_category",
    "SatisfactionScore": "satisfaction_score",
    "MaritalStatus": "marital_status",
    "NumberOfAddress": "address_count",
    "Complain": "complaint_flag",
    "OrderAmountHikeFromlastYear": "order_amount_hike_from_last_year",
    "CouponUsed": "coupon_used_count",
    "OrderCount": "order_count",
    "DaySinceLastOrder": "days_since_last_order",
    "CashbackAmount": "cashback_amount",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logger for readable streaming progress output."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("simulation.data_stream")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Data loading and batch generation helpers
# ---------------------------------------------------------------------------
def load_dataset(file_path: Path = SOURCE_FILE, sheet_name: str = SOURCE_SHEET_NAME) -> pd.DataFrame:
    """Load source dataset from Excel for simulated streaming."""
    if not file_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {file_path}")

    try:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    except ValueError as exc:
        raise ValueError(
            f"Could not read sheet '{sheet_name}' from file '{file_path}'."
        ) from exc


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    """Yield DataFrame slices of size batch_size."""
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def _to_snake_case(name: str) -> str:
    """Convert arbitrary column names into readable snake_case fallback names."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    return name.strip("_").lower()


def rename_columns_informatively(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source column names into informative Bronze table columns."""
    rename_map = {
        col: INFORMATIVE_COLUMN_MAP.get(col, _to_snake_case(col))
        for col in df.columns
    }
    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Streaming workflow
# ---------------------------------------------------------------------------
def stream_data_batches(
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: int = DEFAULT_DELAY_SECONDS,
    bronze_dir: Path = BRONZE_DIR,
) -> None:
    """Stream the source dataset into a Bronze-layer SQLite table."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")

    df = load_dataset()
    bronze_dir.mkdir(parents=True, exist_ok=True)

    total_rows = len(df)
    total_batches = math.ceil(total_rows / batch_size)

    logger.info("Starting data source simulation...")
    logger.info("Total rows in dataset: %d", total_rows)
    logger.info("Batch size: %d", batch_size)
    logger.info("Total batches to stream: %d", total_batches)

    with sqlite3.connect(BRONZE_DB_PATH) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {BRONZE_TABLE_NAME}")
        logger.info("Reset Bronze table: %s", BRONZE_TABLE_NAME)

        for batch_number, batch_df in enumerate(iter_batches(df, batch_size), start=1):
            batch_df = rename_columns_informatively(batch_df.copy())
            batch_df["bronze_batch_id"] = batch_number
            batch_df["bronze_emitted_at_utc"] = pd.Timestamp.utcnow().isoformat()
            batch_df.to_sql(BRONZE_TABLE_NAME, conn, if_exists="append", index=False)

            logger.info(
                "Streaming batch %d/%d -> appended to %s (%d rows)",
                batch_number,
                total_batches,
                BRONZE_TABLE_NAME,
                len(batch_df),
            )

            # Sleep between batches to emulate continuous incoming data.
            if batch_number < total_batches:
                time.sleep(delay_seconds)

    logger.info("Data simulation completed successfully.")
    logger.info("Bronze DB: %s", BRONZE_DB_PATH)
    logger.info("Bronze table: %s", BRONZE_TABLE_NAME)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse optional runtime configuration for batch size and delay."""
    parser = argparse.ArgumentParser(description="Simulate streaming batches into Bronze layer.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rows per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--delay-seconds",
        type=int,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Delay between batches in seconds (default: {DEFAULT_DELAY_SECONDS})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the data streaming simulation from terminal."""
    args = parse_args()
    stream_data_batches(batch_size=args.batch_size, delay_seconds=args.delay_seconds)


if __name__ == "__main__":
    main()
