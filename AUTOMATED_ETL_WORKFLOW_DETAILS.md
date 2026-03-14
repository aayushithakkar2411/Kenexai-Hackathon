# Automated ETL Batch Workflow - Full Step Details

## 1. Objective
Build an automated, production-ready ETL workflow that:
- Accepts new incoming data batches from file drop or database.
- Processes only new/unprocessed batches.
- Triggers ETL processing and warehouse update.
- Triggers ML scoring after ETL.
- Logs and tracks batch lifecycle with timestamps.
- Prevents duplicate reprocessing.

## 2. What Was Implemented

### New scripts
- [create_batch.py](create_batch.py)
- [watcher.py](watcher.py)
- [ml/infer_batch.py](ml/infer_batch.py)

### Updated existing ETL scripts
- [etl/extract.py](etl/extract.py)
- [etl/transform.py](etl/transform.py)

### New data directory
- [data/bronze_branch](data/bronze_branch)

## 3. Architecture and Data Flow

1. create_batch.py
- Reads raw source data from CSV or Excel.
- Samples N rows to simulate a new incoming batch.
- Writes batch CSV to Bronze branch folder atomically.
- Inserts batch rows into Bronze SQLite tables.
- Registers batch in tracking table with status new.

2. watcher.py
- Checks Bronze branch and Bronze DB for pending batches.
- Uses file stability checks to avoid partial-write reads.
- Processes each pending batch through ETL.
- Updates batch state and timestamps during each stage.
- Rebuilds warehouse and runs ML scoring.
- Marks rows processed and batch status processed.

3. ETL
- extract.py now supports CSV and Excel dynamic input.
- transform.py now supports CLI batch args and append mode.
- Silver output includes batch metadata.

4. ML trigger
- infer_batch.py reads only current batch from Silver.
- Scores churn probability using churn_model.pkl.
- Writes results to Gold DB table and latest CSV.

## 4. Bronze Database Tables

Database file:
- data/bronze/bronze_batches.db

Table 1: bronze_batch_registry
- batch_id (primary key)
- source_file
- created_at_utc
- row_count
- status (new, processing, processed, failed)
- processing_started_at_utc
- processed_at_utc
- etl_started_at_utc
- etl_finished_at_utc
- ml_started_at_utc
- ml_finished_at_utc
- error_message

Table 2: bronze_batch_rows
- batch_id
- row_in_batch
- ingest_created_at_utc
- processed (0 or 1)
- plus dynamic source columns from batch data

## 5. Script Details

## 5.1 create_batch.py
Purpose:
- Simulate incoming data batches and register them for processing.

Main behavior:
- Ensures Bronze folder and DB exist.
- Ensures registry table exists.
- Auto-aligns rows table schema with incoming dataset columns.
- Saves CSV with atomic temp-to-final rename.
- Inserts rows and marks processed = 0.
- Adds registry record with status new.

Key command:
- python create_batch.py

Optional arguments:
- --source
- --sheet
- --sample-size
- --seed
- --bronze-dir
- --bronze-db
- --rows-table
- --registry-table

## 5.2 watcher.py
Purpose:
- Detect and process unprocessed batches safely and idempotently.

Main behavior:
- Ensures Bronze registry exists.
- Discovers file batches and DB pending rows.
- Uses stability check to avoid partially written files.
- Triggers ETL script for each batch with batch_id.
- Optional notebook execution via nbconvert.
- Passes BATCH_INPUT_PATH and BATCH_ID env vars for notebook compatibility.
- Rebuilds warehouse.
- Runs ML inference script per batch.
- Marks bronze rows processed and updates status processed.
- Logs each step to file.

Key command:
- python watcher.py

Important arguments:
- --source-mode file|db|both
- --etl-script
- --etl-notebook
- --warehouse-script
- --ml-script
- --stability-seconds
- --log-file

## 5.3 etl/extract.py
Enhancements:
- Supports CSV and Excel.
- Keeps original default source behavior.
- Supports BATCH_INPUT_PATH override for dynamic runs.

## 5.4 etl/transform.py
Enhancements:
- Added CLI args for batch input and append mode.
- Added batch_id and processed_at_utc into Silver output.
- Added append schema compatibility behavior for Silver table.

## 5.5 ml/infer_batch.py
Purpose:
- Score one specific batch after ETL.

Behavior:
- Loads model artifact.
- Pulls only rows matching given batch_id from Silver.
- Builds feature matrix for expected model columns.
- Applies scaler if present.
- Predicts churn probability.
- Writes output to:
  - Gold table: fact_customer_churn_scoring_results
  - CSV: data/gold/latest_batch_predictions.csv

## 6. Run Commands Used

From workspace root:
- e:/Kenexai-Hackathon-2k26/.venv/Scripts/python.exe create_batch.py
- e:/Kenexai-Hackathon-2k26/.venv/Scripts/python.exe watcher.py

## 7. Execution Outcome Confirmation

Workflow run status:
- create_batch.py: Success
- watcher.py: Success
- ETL per batch: Success
- Warehouse build: Success
- ML scoring: Success

Observed successful batch IDs:
- batch_20260314T204855Z_2584c0bd
- batch_20260314T204805Z_eb73b86c

Confirmed outputs:
- Silver contains both batches with 100 rows each.
- Bronze registry shows processed status for 2 batches.
- Bronze rows have processed flag updates.
- Gold prediction table has rows from latest scoring run.

## 8. Logging and Auditing

Primary watcher log:
- [data/logs/watcher.log](data/logs/watcher.log)

Log includes:
- Batch discovery count.
- Batch start/end processing.
- ETL, warehouse, and ML command execution.
- stdout and stderr capture.
- errors with stack traces when failures occur.

## 9. Duplicate Prevention and Safety Controls

Implemented controls:
- Registry status gating: only new or failed are picked.
- Row-level processed flag in bronze_batch_rows.
- File stability check before reading file batches.
- Atomic CSV write in create_batch.py.
- Batch-level state transitions and timestamps.

## 10. Notebook Support

If using notebook ETL in watcher:
- Use --etl-notebook path.
- watcher sets:
  - BATCH_INPUT_PATH
  - BATCH_ID

Notebook requirement:
- Notebook logic should read dynamic input path from environment variables.

## 11. Scheduling Setup

## 11.1 Linux crontab examples
Every 10 minutes (generate batch):
- */10 * * * * cd /path/to/Kenexai-Hackathon-2k26 && /path/to/python create_batch.py >> data/logs/create_batch.log 2>&1

Every 5 minutes (watch and process):
- */5 * * * * cd /path/to/Kenexai-Hackathon-2k26 && /path/to/python watcher.py >> data/logs/watcher_cron.log 2>&1

## 11.2 Windows (current environment)
Use Task Scheduler:
- Task A: run create_batch.py every 10 minutes.
- Task B: run watcher.py every 5 minutes.
- Start in folder should be workspace root.

## 12. Known Behavior and Optional Improvement

Current behavior:
- ML scoring results table may not remain cumulative if warehouse rebuild removes legacy tables including scoring table.

Optional improvement:
- Update [warehouse/build_warehouse.py](warehouse/build_warehouse.py) to stop dropping fact_customer_churn_scoring_results if cumulative history is desired.

## 13. Quick Verification Queries

Bronze status check:
- select status, count(*) from bronze_batch_registry group by status;

Silver batch check:
- select batch_id, count(*) from silver_customer_churn_curated where batch_id is not null group by batch_id;

Prediction row count:
- select count(*) from fact_customer_churn_scoring_results;

## 14. Final Confirmation
The automated workflow is functional and confirmed through live execution in this environment. It is ready for scheduled automation with safe batch handling, ETL triggering, ML scoring, status tracking, and operational logging.
