from __future__ import annotations

import argparse
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "raw" / "E Commerce Dataset.xlsx"
DEFAULT_SOURCE_SHEET = "E Comm"
DEFAULT_BRONZE_DIR = PROJECT_ROOT / "data" / "bronze_branch"
DEFAULT_BRONZE_DB = PROJECT_ROOT / "data" / "bronze" / "bronze_batches.db"
DEFAULT_ROWS_TABLE = "bronze_batch_rows"
DEFAULT_REGISTRY_TABLE = "bronze_batch_registry"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_paths(bronze_dir: Path, db_path: Path) -> None:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_bronze_tables(conn: sqlite3.Connection, rows_table: str, registry_table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {registry_table} (
            batch_id TEXT PRIMARY KEY,
            source_file TEXT,
            created_at_utc TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            processing_started_at_utc TEXT,
            processed_at_utc TEXT,
            etl_started_at_utc TEXT,
            etl_finished_at_utc TEXT,
            ml_started_at_utc TEXT,
            ml_finished_at_utc TEXT,
            error_message TEXT
        )
        """
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def ensure_rows_table_compatible(
    conn: sqlite3.Connection,
    rows_table: str,
    expected_columns: list[str],
) -> None:
    """Ensure rows table contains all expected columns for append writes."""
    if not _table_exists(conn, rows_table):
        return

    info = conn.execute(f"PRAGMA table_info({rows_table})").fetchall()
    existing_columns = {row[1] for row in info}

    for col in expected_columns:
        if col not in existing_columns:
            conn.execute(f'ALTER TABLE {rows_table} ADD COLUMN "{col}" TEXT')


def read_source_dataset(source_path: Path, sheet_name: str) -> pd.DataFrame:
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source_path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported source format '{suffix}'. Use CSV or Excel.")


def save_batch_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def generate_batch_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"batch_{ts}_{uuid.uuid4().hex[:8]}"


def create_batch(
    source_path: Path,
    source_sheet: str,
    sample_size: int,
    random_state: int,
    bronze_dir: Path,
    bronze_db: Path,
    rows_table: str,
    registry_table: str,
) -> tuple[str, Path, int]:
    ensure_paths(bronze_dir, bronze_db)

    df = read_source_dataset(source_path, source_sheet)
    if df.empty:
        raise ValueError("Source dataset is empty; cannot create batch.")

    n = min(sample_size, len(df))
    batch_df = df.sample(n=n, random_state=random_state).reset_index(drop=True)

    batch_id = generate_batch_id()
    created_at = utc_now_iso()
    batch_path = bronze_dir / f"{batch_id}.csv"

    save_batch_csv_atomic(batch_df, batch_path)

    db_df = batch_df.copy()
    db_df.insert(0, "batch_id", batch_id)
    db_df.insert(1, "row_in_batch", range(1, len(db_df) + 1))
    db_df["ingest_created_at_utc"] = created_at
    db_df["processed"] = 0

    with sqlite3.connect(bronze_db) as conn:
        ensure_bronze_tables(conn, rows_table, registry_table)
        ensure_rows_table_compatible(conn, rows_table, db_df.columns.tolist())

        db_df.to_sql(rows_table, conn, if_exists="append", index=False)

        conn.execute(
            f"""
            INSERT OR REPLACE INTO {registry_table}
            (
                batch_id,
                source_file,
                created_at_utc,
                row_count,
                status,
                processing_started_at_utc,
                processed_at_utc,
                etl_started_at_utc,
                etl_finished_at_utc,
                ml_started_at_utc,
                ml_finished_at_utc,
                error_message
            )
            VALUES (?, ?, ?, ?, 'new', NULL, NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            (batch_id, str(batch_path), created_at, len(batch_df)),
        )
        conn.commit()

    return batch_id, batch_path, len(batch_df)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a simulated incoming batch from raw data.")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE_PATH), help="Raw source CSV/Excel path.")
    parser.add_argument("--sheet", type=str, default=DEFAULT_SOURCE_SHEET, help="Excel sheet name for source file.")
    parser.add_argument("--sample-size", type=int, default=100, help="Rows to sample for the new batch.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--bronze-dir", type=str, default=str(DEFAULT_BRONZE_DIR), help="Folder to drop batch CSV files.")
    parser.add_argument("--bronze-db", type=str, default=str(DEFAULT_BRONZE_DB), help="SQLite DB for bronze batches.")
    parser.add_argument("--rows-table", type=str, default=DEFAULT_ROWS_TABLE, help="SQLite table for batch rows.")
    parser.add_argument(
        "--registry-table",
        type=str,
        default=DEFAULT_REGISTRY_TABLE,
        help="SQLite table that tracks batch processing state.",
    )
    return parser


def main() -> None:
    args = build_cli().parse_args()

    batch_id, batch_path, row_count = create_batch(
        source_path=Path(args.source),
        source_sheet=args.sheet,
        sample_size=args.sample_size,
        random_state=args.seed,
        bronze_dir=Path(args.bronze_dir),
        bronze_db=Path(args.bronze_db),
        rows_table=args.rows_table,
        registry_table=args.registry_table,
    )

    print(f"Created batch_id={batch_id}")
    print(f"Saved CSV batch to: {batch_path}")
    print(f"Inserted {row_count} rows into bronze DB")


if __name__ == "__main__":
    main()
