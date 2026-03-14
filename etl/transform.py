"""TRANSFORM stage for customer churn ETL pipeline.

This script loads raw data via extract_data(), applies cleaning and
standardization rules, validates quality, and saves Silver-layer output.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from extract import extract_data


# Silver layer output path (Medallion Architecture)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SILVER_DB_PATH = SILVER_DIR / "silver_layer.db"
SILVER_TABLE_NAME = "silver_customer_churn_curated"

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


# Important columns defined by business/data requirements
NUMERIC_COLUMNS = [
	"Tenure",
	"WarehouseToHome",
	"HourSpendOnApp",
	"NumberOfDeviceRegistered",
	"SatisfactionScore",
	"NumberOfAddress",
	"OrderAmountHikeFromlastYear",
	"CouponUsed",
	"OrderCount",
	"DaySinceLastOrder",
	"CashbackAmount",
]

CATEGORICAL_COLUMNS = [
	"PreferredLoginDevice",
	"PreferredPaymentMode",
	"Gender",
	"PreferedOrderCat",
	"MaritalStatus",
]


def _canonicalize_text(value: object) -> str:
	"""Normalize free-text categorical values for robust matching."""
	if pd.isna(value):
		return ""
	text = str(value).strip().lower()
	text = re.sub(r"\s+", " ", text)
	return text


def standardize_business_categories(df: pd.DataFrame) -> pd.DataFrame:
	"""Standardize synonymous category labels into business-approved canonical values."""
	mappings = {
		"PreferredLoginDevice": {
			"mobile": "Mobile Phone",
			"phone": "Mobile Phone",
			"mobile phone": "Mobile Phone",
		},
		"PreferredPaymentMode": {
			"cc": "Credit Card",
			"credit card": "Credit Card",
			"cod": "Cash on Delivery",
			"cash on delivery": "Cash on Delivery",
		},
		"PreferedOrderCat": {
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


def print_diagnostics(df: pd.DataFrame, title: str) -> None:
	"""Print concise diagnostics for ETL monitoring."""
	print(f"\n--- {title} ---")
	print(f"Shape: {df.shape}")
	print("Dtypes:")
	print(df.dtypes)
	print("Missing values per column:")
	print(df.isnull().sum())


def coerce_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
	"""Ensure configured numeric columns are numeric."""
	for col in numeric_columns:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Fill missing values: median for numeric, mode for categorical."""
	for col in NUMERIC_COLUMNS:
		if col in df.columns:
			median_value = df[col].median()
			df[col] = df[col].fillna(median_value)

	for col in CATEGORICAL_COLUMNS:
		if col in df.columns:
			mode_series = df[col].mode(dropna=True)
			if not mode_series.empty:
				df[col] = df[col].fillna(mode_series.iloc[0])

	return df


def apply_basic_outlier_filter(df: pd.DataFrame) -> pd.DataFrame:
	"""Remove unrealistic records based on rule-based thresholds."""
	if "OrderCount" in df.columns:
		df = df[df["OrderCount"] <= 30]
	if "HourSpendOnApp" in df.columns:
		df = df[df["HourSpendOnApp"] <= 15]
	return df


def validate_cleaned_dataset(df: pd.DataFrame) -> None:
	"""Validate cleaned data quality before save."""
	total_null_values = int(df.isnull().sum().sum())
	if total_null_values != 0:
		raise ValueError(
			f"Validation failed: cleaned dataset still has {total_null_values} null values."
		)


def _to_snake_case(name: str) -> str:
	"""Convert arbitrary column name into a readable snake_case fallback."""
	name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
	name = re.sub(r"[^A-Za-z0-9]+", "_", name)
	return name.strip("_").lower()


def rename_columns_informatively(df: pd.DataFrame) -> pd.DataFrame:
	"""Rename columns to business-readable warehouse-friendly names."""
	rename_map = {
		col: INFORMATIVE_COLUMN_MAP.get(col, _to_snake_case(col))
		for col in df.columns
	}
	return df.rename(columns=rename_map)


def save_to_silver_db(df: pd.DataFrame, batch_id: str | None = None, append: bool = False) -> None:
	"""Persist cleaned Silver dataset into SQLite with optional batch metadata."""
	SILVER_DIR.mkdir(parents=True, exist_ok=True)
	df_db = rename_columns_informatively(df)
	if batch_id:
		df_db["batch_id"] = batch_id
	df_db["processed_at_utc"] = datetime.now(timezone.utc).isoformat()

	write_mode = "append" if append else "replace"
	with sqlite3.connect(SILVER_DB_PATH) as conn:
		if write_mode == "append":
			table_exists = conn.execute(
				"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
				(SILVER_TABLE_NAME,),
			).fetchone()
			if table_exists:
				existing_cols = {
					row[1] for row in conn.execute(f"PRAGMA table_info({SILVER_TABLE_NAME})").fetchall()
				}
				for col in df_db.columns:
					if col not in existing_cols:
						conn.execute(f'ALTER TABLE {SILVER_TABLE_NAME} ADD COLUMN "{col}" TEXT')
		df_db.to_sql(SILVER_TABLE_NAME, conn, if_exists=write_mode, index=False)


def clean_data(
	source_path: Path | None = None,
	source_sheet: str = "E Comm",
	batch_id: str | None = None,
	append_to_silver: bool = False,
) -> pd.DataFrame:
	"""Main transform workflow to produce Silver-layer cleaned dataset."""
	# 1) Load extracted dataset from EXTRACT stage
	df = extract_data(file_path=source_path, sheet_name=source_sheet)

	# 2) Print initial diagnostics
	print_diagnostics(df, "Initial Dataset Diagnostics")

	# 3) Remove duplicate rows
	before_dupes = len(df)
	df = df.drop_duplicates().copy()
	removed_dupes = before_dupes - len(df)
	print(f"\nRemoved duplicate rows: {removed_dupes}")

	# 4 & 5) Fix numeric dtypes and fill missing values
	df = coerce_numeric_columns(df, NUMERIC_COLUMNS)
	df = standardize_business_categories(df)
	df = fill_missing_values(df)

	# 6) Outlier filtering
	before_filter = len(df)
	df = apply_basic_outlier_filter(df)
	filtered_rows = before_filter - len(df)
	print(f"Rows removed by outlier filtering: {filtered_rows}")

	# Optional step: keep CustomerID for future warehouse design
	# 7) Drop unnecessary columns only if needed (not dropping by default)

	# 8) Validate cleaned dataset quality
	validate_cleaned_dataset(df)

	# 9) Save Silver layer output as SQLite table
	save_to_silver_db(df, batch_id=batch_id, append=append_to_silver)

	# Final diagnostics and confirmation message
	print_diagnostics(df, "Cleaned Dataset Diagnostics")
	print("\nCleaned dataset saved successfully.")
	print(f"Rows: {df.shape[0]}")
	print(f"Columns: {df.shape[1]}")
	print(f"Saved to DB: {SILVER_DB_PATH}")
	print(f"Table name: {SILVER_TABLE_NAME}")
	if batch_id:
		print(f"Batch ID: {batch_id}")

	return df


def _build_cli() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run ETL TRANSFORM stage for raw or batch input.")
	parser.add_argument(
		"--input-path",
		type=str,
		default=None,
		help="Optional CSV/Excel path for batch input.",
	)
	parser.add_argument(
		"--input-sheet",
		type=str,
		default="E Comm",
		help="Excel sheet name when --input-path points to .xlsx/.xls.",
	)
	parser.add_argument(
		"--batch-id",
		type=str,
		default=None,
		help="Optional batch identifier to persist alongside Silver data.",
	)
	parser.add_argument(
		"--append",
		action="store_true",
		help="Append to Silver table instead of replacing it.",
	)
	return parser


if __name__ == "__main__":
	args = _build_cli().parse_args()
	input_path = Path(args.input_path) if args.input_path else None
	clean_data(
		source_path=input_path,
		source_sheet=args.input_sheet,
		batch_id=args.batch_id,
		append_to_silver=args.append,
	)
