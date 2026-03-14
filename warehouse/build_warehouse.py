"""Build a Gold-layer Star Schema warehouse from Silver-layer cleaned data.

Medallion flow:
- Bronze: raw files
- Silver: cleaned dataset
- Gold: analytics-ready dimensional model in SQLite

This script creates all required dimensions and the fact table, validates
referential integrity, and persists tables to data/gold/warehouse.db.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DB_PATH = PROJECT_ROOT / "data" / "silver" / "silver_layer.db"
SILVER_TABLE_NAME = "silver_customer_churn_curated"
WAREHOUSE_DB_PATH = PROJECT_ROOT / "data" / "gold" / "warehouse.db"

# Gold table names
DIM_CUSTOMER_TABLE = "dim_customer_profile"
DIM_DEVICE_TABLE = "dim_login_device"
DIM_PAYMENT_TABLE = "dim_payment_method"
DIM_ORDER_CATEGORY_TABLE = "dim_order_category"
DIM_TIME_TABLE = "dim_customer_time_window"
FACT_TABLE_NAME = "fact_customer_churn_metrics"
LEGACY_TABLES = [
    "dim_customer",
    "dim_device",
    "dim_payment",
    "dim_time",
    "fact_customer_behavior",
    "fact_customer_churn_events_stream",
    "fact_customer_churn_scoring_results",
    "agg_customer_churn_hourly_summary",
]

# Dimension attribute definitions (Silver schema already informative)
CUSTOMER_COLUMNS = ["customer_id", "customer_gender", "marital_status", "city_tier"]
DEVICE_COLUMNS = ["preferred_login_device"]
PAYMENT_COLUMNS = ["preferred_payment_mode"]
ORDER_CATEGORY_COLUMNS = ["preferred_order_category"]
TIME_COLUMNS = ["days_since_last_order", "customer_tenure_months"]

# Fact measures required for analytical queries
FACT_MEASURE_COLUMNS = [
    "customer_tenure_months",
    "warehouse_to_home_distance",
    "hours_spent_on_app",
    "registered_device_count",
    "satisfaction_score",
    "address_count",
    "complaint_flag",
    "order_amount_hike_from_last_year",
    "coupon_used_count",
    "order_count",
    "days_since_last_order",
    "cashback_amount",
    "churn_flag",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return logger for warehouse build workflow."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("warehouse.build_warehouse")


logger = configure_logging()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ensure_columns_exist(df: pd.DataFrame, required_columns: list[str], context: str) -> None:
    """Validate required columns exist before transformation."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}")


def load_cleaned_dataset(
    silver_db_path: Path = SILVER_DB_PATH,
    silver_table_name: str = SILVER_TABLE_NAME,
) -> pd.DataFrame:
    """Load Silver-layer cleaned dataset from SQLite."""
    if not silver_db_path.exists():
        raise FileNotFoundError(f"Silver DB not found: {silver_db_path}")

    with sqlite3.connect(silver_db_path) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            conn,
            params=(silver_table_name,),
        )
        if tables.empty:
            raise ValueError(f"Silver table not found: {silver_table_name}")

        df = pd.read_sql_query(f"SELECT * FROM {silver_table_name}", conn)

    logger.info("Loaded cleaned dataset from %s (%s)", silver_db_path, silver_table_name)
    logger.info("Input dataset shape: rows=%d, cols=%d", *df.shape)
    return df


def _canonicalize_text(value: object) -> str:
    """Normalize free-text categorical values for robust matching."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def standardize_business_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical naming for key business dimensions before modeling."""
    mappings = {
        "preferred_login_device": {
            "mobile": "Mobile Phone",
            "phone": "Mobile Phone",
            "mobile phone": "Mobile Phone",
        },
        "preferred_payment_mode": {
            "cc": "Credit Card",
            "credit card": "Credit Card",
            "cod": "Cash on Delivery",
            "cash on delivery": "Cash on Delivery",
        },
        "preferred_order_category": {
            "mobile": "Mobile Phone",
            "mobile phone": "Mobile Phone",
        },
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda value: mapping.get(_canonicalize_text(value), value)
            )

    return df


def create_dimension_table(
    source_df: pd.DataFrame,
    natural_key_columns: list[str],
    surrogate_key_column: str,
) -> pd.DataFrame:
    """Create a dimension table by deduplicating natural key attributes."""
    ensure_columns_exist(source_df, natural_key_columns, f"dimension {surrogate_key_column}")

    dim_df = source_df[natural_key_columns].drop_duplicates().reset_index(drop=True)
    dim_df.insert(0, surrogate_key_column, range(1, len(dim_df) + 1))
    return dim_df


def validate_dimension_table(dim_df: pd.DataFrame, surrogate_key_column: str, table_name: str) -> None:
    """Validate dimension table primary-key quality constraints."""
    if dim_df[surrogate_key_column].duplicated().any():
        raise ValueError(f"Duplicate primary keys found in {table_name}.{surrogate_key_column}")

    if dim_df[surrogate_key_column].isnull().any():
        raise ValueError(f"Null primary keys found in {table_name}.{surrogate_key_column}")

    if dim_df.isnull().any().any():
        null_counts = dim_df.isnull().sum()
        raise ValueError(
            f"Null values found in {table_name}: "
            + null_counts[null_counts > 0].to_dict().__repr__()
        )


def validate_foreign_keys(
    fact_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    foreign_key: str,
    dim_table_name: str,
) -> None:
    """Validate that every fact FK value exists in the related dimension PK."""
    valid_values = set(dim_df[foreign_key].tolist())
    invalid_mask = ~fact_df[foreign_key].isin(valid_values)
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        raise ValueError(
            f"Invalid foreign keys in {FACT_TABLE_NAME}.{foreign_key} "
            f"not found in {dim_table_name}.{foreign_key}: {invalid_count}"
        )


def build_star_schema_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build star schema dimensions and fact table from cleaned source data."""
    df = standardize_business_categories(df.copy())
    logger.info("Building dimension tables.")

    # Build dimensions with surrogate keys.
    dim_customer = create_dimension_table(df, CUSTOMER_COLUMNS, "customer_profile_key")
    dim_device = create_dimension_table(df, DEVICE_COLUMNS, "login_device_key")
    dim_payment = create_dimension_table(df, PAYMENT_COLUMNS, "payment_method_key")
    dim_order_category = create_dimension_table(df, ORDER_CATEGORY_COLUMNS, "order_category_key")

    # Optional time dimension: created when source columns are available.
    time_dimension_enabled = all(col in df.columns for col in TIME_COLUMNS)
    dim_time = pd.DataFrame()
    if time_dimension_enabled:
        dim_time = create_dimension_table(df, TIME_COLUMNS, "customer_time_key")

    validate_dimension_table(dim_customer, "customer_profile_key", DIM_CUSTOMER_TABLE)
    validate_dimension_table(dim_device, "login_device_key", DIM_DEVICE_TABLE)
    validate_dimension_table(dim_payment, "payment_method_key", DIM_PAYMENT_TABLE)
    validate_dimension_table(dim_order_category, "order_category_key", DIM_ORDER_CATEGORY_TABLE)
    if time_dimension_enabled:
        validate_dimension_table(dim_time, "customer_time_key", DIM_TIME_TABLE)

    logger.info("Created %s with %d rows", DIM_CUSTOMER_TABLE, len(dim_customer))
    logger.info("Created %s with %d rows", DIM_DEVICE_TABLE, len(dim_device))
    logger.info("Created %s with %d rows", DIM_PAYMENT_TABLE, len(dim_payment))
    logger.info("Created %s with %d rows", DIM_ORDER_CATEGORY_TABLE, len(dim_order_category))
    if time_dimension_enabled:
        logger.info("Created %s with %d rows", DIM_TIME_TABLE, len(dim_time))

    logger.info("Building fact table with surrogate-key references.")

    # Build fact table by replacing dimension attributes with surrogate keys.
    fact_df = df.copy()
    fact_df = fact_df.merge(dim_customer, on=CUSTOMER_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_device, on=DEVICE_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_payment, on=PAYMENT_COLUMNS, how="left")
    fact_df = fact_df.merge(dim_order_category, on=ORDER_CATEGORY_COLUMNS, how="left")

    foreign_key_columns = [
        "customer_profile_key",
        "login_device_key",
        "payment_method_key",
        "order_category_key",
    ]

    if time_dimension_enabled:
        fact_df = fact_df.merge(dim_time, on=TIME_COLUMNS, how="left")
        foreign_key_columns.append("customer_time_key")

    ensure_columns_exist(fact_df, FACT_MEASURE_COLUMNS, "fact table measures")

    fact_columns = foreign_key_columns + FACT_MEASURE_COLUMNS
    fact_customer_churn_metrics = fact_df[fact_columns].copy()

    # Validate referential integrity assumptions before persistence.
    if len(fact_customer_churn_metrics) != len(df):
        raise ValueError(
            "Fact row count does not match input row count. "
            f"fact={len(fact_customer_churn_metrics)}, input={len(df)}"
        )

    null_fk_counts = fact_customer_churn_metrics[foreign_key_columns].isnull().sum()
    if (null_fk_counts > 0).any():
        raise ValueError(
            "Null foreign keys detected in fact table: "
            + null_fk_counts[null_fk_counts > 0].to_dict().__repr__()
        )

    # Explicit FK validity checks against every dimension table.
    validate_foreign_keys(
        fact_customer_churn_metrics,
        dim_customer,
        "customer_profile_key",
        DIM_CUSTOMER_TABLE,
    )
    validate_foreign_keys(
        fact_customer_churn_metrics,
        dim_device,
        "login_device_key",
        DIM_DEVICE_TABLE,
    )
    validate_foreign_keys(
        fact_customer_churn_metrics,
        dim_payment,
        "payment_method_key",
        DIM_PAYMENT_TABLE,
    )
    validate_foreign_keys(
        fact_customer_churn_metrics,
        dim_order_category,
        "order_category_key",
        DIM_ORDER_CATEGORY_TABLE,
    )
    if time_dimension_enabled:
        validate_foreign_keys(
            fact_customer_churn_metrics,
            dim_time,
            "customer_time_key",
            DIM_TIME_TABLE,
        )

    logger.info("Validation passed: fact row count and foreign keys are valid.")

    logger.info("Created %s with %d rows", FACT_TABLE_NAME, len(fact_customer_churn_metrics))

    tables: Dict[str, pd.DataFrame] = {
        DIM_CUSTOMER_TABLE: dim_customer,
        DIM_DEVICE_TABLE: dim_device,
        DIM_PAYMENT_TABLE: dim_payment,
        DIM_ORDER_CATEGORY_TABLE: dim_order_category,
        FACT_TABLE_NAME: fact_customer_churn_metrics,
    }

    if time_dimension_enabled:
        tables[DIM_TIME_TABLE] = dim_time

    return tables


def save_tables_to_sqlite(tables: Dict[str, pd.DataFrame], db_path: Path = WAREHOUSE_DB_PATH) -> None:
    """Persist all warehouse tables into SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving Star Schema tables to SQLite database.")

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        # Remove stale legacy schema tables so Gold stays consistently meaningful.
        for legacy_table in LEGACY_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {legacy_table}")

        for table_name, table_df in tables.items():
            table_df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("Saved %s to SQLite with %d rows", table_name, len(table_df))

    logger.info("Warehouse database saved at %s", db_path)


def build_data_warehouse() -> None:
    """Orchestrate end-to-end warehouse build from Silver to Gold."""
    logger.info("Starting warehouse build process (Silver -> Gold).")

    df = load_cleaned_dataset()
    tables = build_star_schema_tables(df)
    save_tables_to_sqlite(tables)

    logger.info("Warehouse build completed successfully.")


if __name__ == "__main__":
    build_data_warehouse()
