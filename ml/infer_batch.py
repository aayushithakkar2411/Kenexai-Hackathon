from __future__ import annotations

import argparse
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "ml" / "churn_model.pkl"
DEFAULT_SILVER_DB = PROJECT_ROOT / "data" / "silver" / "silver_layer.db"
DEFAULT_SILVER_TABLE = "silver_customer_churn_curated"
DEFAULT_OUTPUT_DB = PROJECT_ROOT / "data" / "gold" / "warehouse.db"
DEFAULT_OUTPUT_TABLE = "fact_customer_churn_scoring_results"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "gold" / "latest_batch_predictions.csv"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")

    with path.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise TypeError("Expected model artifact to be a dict.")

    if "model" not in artifact or "feature_columns" not in artifact:
        raise KeyError("Model artifact requires keys: model, feature_columns")

    return artifact


def load_batch_from_silver(silver_db: Path, silver_table: str, batch_id: str) -> pd.DataFrame:
    if not silver_db.exists():
        raise FileNotFoundError(f"Silver DB not found: {silver_db}")

    query = f"SELECT * FROM {silver_table} WHERE batch_id = ?"
    with sqlite3.connect(silver_db) as conn:
        df = pd.read_sql_query(query, conn, params=(batch_id,))

    if df.empty:
        raise ValueError(f"No Silver rows found for batch_id={batch_id}")

    return df


def build_model_input(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    # Start from all-zero feature frame, then fill only known features.
    x = pd.DataFrame(0.0, index=df.index, columns=feature_columns)

    for col in feature_columns:
        if col in df.columns:
            x[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return x


def score_batch(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_columns = list(artifact["feature_columns"])

    x = build_model_input(df, feature_columns)
    x_eval = x

    if scaler is not None:
        x_eval = pd.DataFrame(
            scaler.transform(x),
            columns=x.columns,
            index=x.index,
        )

    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(x_eval)
    churn_proba = np.clip(proba[:, 1], 0.0, 1.0)

    out = pd.DataFrame(
        {
            "batch_id": df["batch_id"].astype(str),
            "row_number": np.arange(1, len(df) + 1),
            "predicted_churn_probability": churn_proba,
            "predicted_churn_flag": (churn_proba >= 0.5).astype(int),
            "scored_at_utc": utc_now_iso(),
        }
    )

    if "customer_id" in df.columns:
        out["customer_id"] = df["customer_id"]

    return out


def persist_predictions(df: pd.DataFrame, output_db: Path, output_table: str, output_csv: Path) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(output_db) as conn:
        df.to_sql(output_table, conn, if_exists="append", index=False)

    df.to_csv(output_csv, index=False)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run batch inference from Silver-layer rows.")
    parser.add_argument("--batch-id", required=True, type=str, help="Batch ID to score from Silver DB")
    parser.add_argument("--artifact", type=str, default=str(DEFAULT_ARTIFACT_PATH), help="Path to model artifact .pkl")
    parser.add_argument("--silver-db", type=str, default=str(DEFAULT_SILVER_DB), help="Path to Silver SQLite DB")
    parser.add_argument("--silver-table", type=str, default=DEFAULT_SILVER_TABLE, help="Silver table name")
    parser.add_argument("--output-db", type=str, default=str(DEFAULT_OUTPUT_DB), help="Output SQLite DB for predictions")
    parser.add_argument("--output-table", type=str, default=DEFAULT_OUTPUT_TABLE, help="Output table for predictions")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path for latest batch predictions")
    return parser


def main() -> None:
    args = build_cli().parse_args()

    artifact = load_artifact(Path(args.artifact))
    batch_df = load_batch_from_silver(Path(args.silver_db), args.silver_table, args.batch_id)
    scored_df = score_batch(batch_df, artifact)
    persist_predictions(
        scored_df,
        output_db=Path(args.output_db),
        output_table=args.output_table,
        output_csv=Path(args.output_csv),
    )

    print(f"Scored batch_id={args.batch_id}")
    print(f"Prediction rows: {len(scored_df)}")
    print(f"Saved latest CSV: {args.output_csv}")
    print(f"Saved DB table: {args.output_table}")


if __name__ == "__main__":
    main()
