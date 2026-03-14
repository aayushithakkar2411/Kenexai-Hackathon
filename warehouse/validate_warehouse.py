"""Validate Star Schema warehouse quality checks in SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path


WAREHOUSE_DB = Path("data/gold/warehouse.db")
SILVER_DB = Path("data/silver/silver_layer.db")
SILVER_TABLE = "silver_customer_churn_curated"
FACT_TABLE = "fact_customer_churn_metrics"

DIMENSION_KEY_MAP = {
    "dim_customer_profile": "customer_profile_key",
    "dim_login_device": "login_device_key",
    "dim_payment_method": "payment_method_key",
    "dim_order_category": "order_category_key",
    "dim_customer_time_window": "customer_time_key",
}


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if a table exists in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone()[0] == 1


def validate_dimension(conn: sqlite3.Connection, table_name: str, key_name: str) -> bool:
    """Check dimension surrogate key uniqueness and null constraints."""
    if not table_exists(conn, table_name):
        print(f"[WARN] {table_name} missing. Skipping.")
        return False

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) - COUNT(DISTINCT {key_name}) FROM {table_name}")
    dup_count = cursor.fetchone()[0]

    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {key_name} IS NULL")
    null_count = cursor.fetchone()[0]

    if dup_count == 0 and null_count == 0:
        print(f"[PASS] {table_name} ({key_name}): unique and no nulls ({total_rows} rows)")
        return True

    print(f"[FAIL] {table_name} ({key_name}): duplicates={dup_count}, nulls={null_count}")
    return False


def validate_fact(conn: sqlite3.Connection, fact_table: str, active_dims: dict[str, str]) -> bool:
    """Validate fact table row count and FK integrity against dimensions."""
    if not table_exists(conn, fact_table):
        print(f"[FAIL] Missing fact table: {fact_table}")
        return False

    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {fact_table}")
    fact_rows = cursor.fetchone()[0]
    print(f"[INFO] {fact_table} row count = {fact_rows}")

    all_passed = True

    if SILVER_DB.exists():
        with sqlite3.connect(SILVER_DB) as silver_conn:
            silver_rows = silver_conn.execute(
                f"SELECT COUNT(*) FROM {SILVER_TABLE}"
            ).fetchone()[0]
        if fact_rows == silver_rows:
            print(f"[PASS] Fact row count matches Silver table ({silver_rows})")
        else:
            print(f"[FAIL] Fact row count mismatch: fact={fact_rows}, silver={silver_rows}")
            all_passed = False

    for dim_table, fk_name in active_dims.items():
        cursor.execute(
            f"SELECT COUNT(*) FROM {fact_table} WHERE {fk_name} IS NULL"
        )
        null_fk = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT COUNT(*) FROM {fact_table} "
            f"WHERE {fk_name} NOT IN (SELECT {fk_name} FROM {dim_table})"
        )
        invalid_fk = cursor.fetchone()[0]

        if null_fk == 0 and invalid_fk == 0:
            print(f"[PASS] FK {fk_name} -> {dim_table}.{fk_name} valid")
        else:
            print(
                f"[FAIL] FK {fk_name} -> {dim_table}.{fk_name} "
                f"nulls={null_fk}, invalid_refs={invalid_fk}"
            )
            all_passed = False

    return all_passed


def main() -> None:
    if not WAREHOUSE_DB.exists():
        print(f"Error: Warehouse DB not found at {WAREHOUSE_DB}")
        return

    conn = sqlite3.connect(WAREHOUSE_DB)
    try:
        print("==== VALIDATING DIMENSIONS ====")
        active_dims: dict[str, str] = {}
        dim_ok = True

        for dim_table, key_name in DIMENSION_KEY_MAP.items():
            if table_exists(conn, dim_table):
                active_dims[dim_table] = key_name
                dim_ok = validate_dimension(conn, dim_table, key_name) and dim_ok
            else:
                print(f"[INFO] {dim_table} not present (optional or not built).")

        print("\n==== VALIDATING FACT TABLE ====")
        fact_ok = validate_fact(conn, FACT_TABLE, active_dims)

        if dim_ok and fact_ok:
            print("\nWarehouse validation completed successfully.")
        else:
            print("\nWarehouse validation completed with failures.")
            raise SystemExit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()