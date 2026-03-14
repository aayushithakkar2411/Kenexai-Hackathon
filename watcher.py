from __future__ import annotations

import argparse
import os
import logging
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BRONZE_DIR = PROJECT_ROOT / "data" / "bronze_branch"
DEFAULT_BRONZE_DB = PROJECT_ROOT / "data" / "bronze" / "bronze_batches.db"
DEFAULT_ROWS_TABLE = "bronze_batch_rows"
DEFAULT_REGISTRY_TABLE = "bronze_batch_registry"
DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "watcher.log"
DEFAULT_ETL_SCRIPT = PROJECT_ROOT / "etl" / "transform.py"
DEFAULT_WAREHOUSE_SCRIPT = PROJECT_ROOT / "warehouse" / "build_warehouse.py"
DEFAULT_ML_SCRIPT = PROJECT_ROOT / "ml" / "infer_batch.py"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline.watcher")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def ensure_resources(bronze_dir: Path, bronze_db: Path, rows_table: str, registry_table: str) -> None:
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bronze_db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(bronze_db) as conn:
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

        conn.commit()


def file_is_stable(path: Path, stability_seconds: int) -> bool:
    if not path.exists() or path.suffix.lower() != ".csv" or path.name.endswith(".tmp"):
        return False

    stat_1 = path.stat()
    time.sleep(stability_seconds)
    if not path.exists():
        return False
    stat_2 = path.stat()

    return stat_1.st_size == stat_2.st_size and stat_1.st_mtime == stat_2.st_mtime


def register_file_batches(bronze_dir: Path, bronze_db: Path, registry_table: str, stability_seconds: int) -> int:
    discovered = 0
    with sqlite3.connect(bronze_db) as conn:
        for csv_path in sorted(bronze_dir.glob("*.csv")):
            if not file_is_stable(csv_path, stability_seconds=stability_seconds):
                continue

            batch_id = csv_path.stem
            row_count = max(sum(1 for _ in csv_path.open("r", encoding="utf-8")) - 1, 0)
            created_at = utc_now_iso()

            conn.execute(
                f"""
                INSERT OR IGNORE INTO {registry_table}
                (batch_id, source_file, created_at_utc, row_count, status)
                VALUES (?, ?, ?, ?, 'new')
                """,
                (batch_id, str(csv_path), created_at, row_count),
            )
            discovered += 1

        conn.commit()
    return discovered


def fetch_pending_batches(bronze_db: Path, registry_table: str) -> list[dict[str, str]]:
    with sqlite3.connect(bronze_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT batch_id, source_file, created_at_utc, status
            FROM {registry_table}
            WHERE status IN ('new', 'failed')
            ORDER BY created_at_utc ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def update_registry_status(
    bronze_db: Path,
    registry_table: str,
    batch_id: str,
    status: str,
    error_message: str | None = None,
    set_processing_started: bool = False,
    set_processed: bool = False,
    set_etl_started: bool = False,
    set_etl_finished: bool = False,
    set_ml_started: bool = False,
    set_ml_finished: bool = False,
) -> None:
    now = utc_now_iso()

    clauses = ["status = ?", "error_message = ?"]
    values: list[object] = [status, error_message]

    if set_processing_started:
        clauses.append("processing_started_at_utc = ?")
        values.append(now)
    if set_processed:
        clauses.append("processed_at_utc = ?")
        values.append(now)
    if set_etl_started:
        clauses.append("etl_started_at_utc = ?")
        values.append(now)
    if set_etl_finished:
        clauses.append("etl_finished_at_utc = ?")
        values.append(now)
    if set_ml_started:
        clauses.append("ml_started_at_utc = ?")
        values.append(now)
    if set_ml_finished:
        clauses.append("ml_finished_at_utc = ?")
        values.append(now)

    values.append(batch_id)

    with sqlite3.connect(bronze_db) as conn:
        conn.execute(
            f"UPDATE {registry_table} SET {', '.join(clauses)} WHERE batch_id = ?",
            values,
        )
        conn.commit()


def materialize_unprocessed_rows_to_csv(
    bronze_db: Path,
    rows_table: str,
    batch_id: str,
    runtime_dir: Path,
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(bronze_db) as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM {rows_table} WHERE batch_id = ? AND processed = 0 ORDER BY row_in_batch",
            conn,
            params=(batch_id,),
        )

    if df.empty:
        raise ValueError(f"No unprocessed DB rows available for batch_id={batch_id}")

    drop_columns = [col for col in ["processed", "ingest_created_at_utc"] if col in df.columns]
    data_columns = [col for col in df.columns if col not in drop_columns]
    df = df[data_columns]

    csv_path = runtime_dir / f"{batch_id}.csv"
    temp_path = csv_path.with_suffix(".csv.tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(csv_path)
    return csv_path


def run_subprocess(
    command: list[str],
    logger: logging.Logger,
    env_overrides: dict[str, str] | None = None,
) -> None:
    logger.info("Running command: %s", " ".join(command))
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    completed = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    if completed.stdout:
        logger.info("stdout: %s", completed.stdout.strip())
    if completed.stderr:
        logger.warning("stderr: %s", completed.stderr.strip())
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")


def mark_batch_rows_processed(bronze_db: Path, rows_table: str, batch_id: str) -> None:
    with sqlite3.connect(bronze_db) as conn:
        conn.execute(
            f"UPDATE {rows_table} SET processed = 1 WHERE batch_id = ?",
            (batch_id,),
        )
        conn.commit()


def process_batch(
    batch: dict[str, str],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    batch_id = batch["batch_id"]
    source_file = Path(batch["source_file"]) if batch.get("source_file") else None

    logger.info("Processing batch: %s", batch_id)
    update_registry_status(
        args.bronze_db,
        args.registry_table,
        batch_id,
        status="processing",
        error_message=None,
        set_processing_started=True,
    )

    try:
        input_path: Path
        if args.source_mode in {"file", "both"} and source_file and source_file.exists():
            if not file_is_stable(source_file, args.stability_seconds):
                raise RuntimeError(f"Batch file still being written: {source_file}")
            input_path = source_file
        elif args.source_mode in {"db", "both"}:
            runtime_dir = PROJECT_ROOT / "data" / "bronze_branch" / ".runtime"
            input_path = materialize_unprocessed_rows_to_csv(
                bronze_db=args.bronze_db,
                rows_table=args.rows_table,
                batch_id=batch_id,
                runtime_dir=runtime_dir,
            )
        else:
            raise RuntimeError("No valid source available for batch processing.")

        update_registry_status(
            args.bronze_db,
            args.registry_table,
            batch_id,
            status="processing",
            error_message=None,
            set_etl_started=True,
        )

        if args.etl_script:
            run_subprocess(
                [
                    sys.executable,
                    args.etl_script,
                    "--input-path",
                    str(input_path),
                    "--batch-id",
                    batch_id,
                    "--append",
                ],
                logger,
            )

        if args.etl_notebook:
            run_subprocess(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    args.etl_notebook,
                    "--ExecutePreprocessor.timeout=1800",
                ],
                logger,
                env_overrides={
                    "BATCH_INPUT_PATH": str(input_path),
                    "BATCH_ID": batch_id,
                },
            )

        update_registry_status(
            args.bronze_db,
            args.registry_table,
            batch_id,
            status="processing",
            error_message=None,
            set_etl_finished=True,
        )

        if args.warehouse_script:
            run_subprocess([sys.executable, args.warehouse_script], logger)

        if args.ml_script:
            update_registry_status(
                args.bronze_db,
                args.registry_table,
                batch_id,
                status="processing",
                error_message=None,
                set_ml_started=True,
            )
            run_subprocess([sys.executable, args.ml_script, "--batch-id", batch_id], logger)
            update_registry_status(
                args.bronze_db,
                args.registry_table,
                batch_id,
                status="processing",
                error_message=None,
                set_ml_finished=True,
            )

        mark_batch_rows_processed(args.bronze_db, args.rows_table, batch_id)

        update_registry_status(
            args.bronze_db,
            args.registry_table,
            batch_id,
            status="processed",
            error_message=None,
            set_processed=True,
        )
        logger.info("Batch processed successfully: %s", batch_id)

    except Exception as exc:  # noqa: BLE001
        update_registry_status(
            args.bronze_db,
            args.registry_table,
            batch_id,
            status="failed",
            error_message=str(exc),
        )
        logger.exception("Batch failed: %s", batch_id)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch and process unprocessed ETL batches.")
    parser.add_argument("--bronze-dir", type=Path, default=DEFAULT_BRONZE_DIR, help="Bronze batch folder")
    parser.add_argument("--bronze-db", type=Path, default=DEFAULT_BRONZE_DB, help="Bronze SQLite DB path")
    parser.add_argument("--rows-table", type=str, default=DEFAULT_ROWS_TABLE, help="Bronze rows table")
    parser.add_argument("--registry-table", type=str, default=DEFAULT_REGISTRY_TABLE, help="Batch registry table")
    parser.add_argument("--source-mode", choices=["file", "db", "both"], default="both", help="Batch discovery source")
    parser.add_argument("--stability-seconds", type=int, default=2, help="File stability wait to avoid partial writes")
    parser.add_argument("--etl-script", type=str, default=str(DEFAULT_ETL_SCRIPT), help="ETL Python script path")
    parser.add_argument("--etl-notebook", type=str, default="", help="Optional ETL notebook to execute with nbconvert")
    parser.add_argument("--warehouse-script", type=str, default=str(DEFAULT_WAREHOUSE_SCRIPT), help="Warehouse build script path")
    parser.add_argument("--ml-script", type=str, default=str(DEFAULT_ML_SCRIPT), help="ML inference script path")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Watcher log file")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    logger = build_logger(args.log_file)

    ensure_resources(args.bronze_dir, args.bronze_db, args.rows_table, args.registry_table)

    discovered = 0
    if args.source_mode in {"file", "both"}:
        discovered = register_file_batches(
            args.bronze_dir,
            args.bronze_db,
            args.registry_table,
            args.stability_seconds,
        )

    pending_batches = fetch_pending_batches(args.bronze_db, args.registry_table)
    logger.info("Discovered file batches this run: %d", discovered)
    logger.info("Pending batches: %d", len(pending_batches))

    for batch in pending_batches:
        process_batch(batch, args, logger)

    logger.info("Watcher run completed.")


if __name__ == "__main__":
    main()
