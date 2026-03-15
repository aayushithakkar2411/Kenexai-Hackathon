from __future__ import annotations

from pathlib import Path
import sqlite3
import subprocess
import sys
import socket
import time
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
	page_title="Churn Intelligence Command Center",
	page_icon="📊",
	layout="wide",
)


THEME = {
	"bg": "#f7f7f2",
	"ink": "#202124",
	"muted": "#5f6368",
	"accent": "#0f766e",
	"accent_alt": "#d97706",
	"card": "#ffffff",
	"line": "#d6d3d1",
	"danger": "#b91c1c",
}


CITY_TIER_NAME_MAP = {
	1: "Metro City",
	2: "Urban City",
	3: "Growth City",
}


def _project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _predict_app_path() -> Path:
	return _project_root() / "frontend" / "ml predict" / "predict.py"


def _is_port_open(port: int) -> bool:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.settimeout(0.4)
		return sock.connect_ex(("127.0.0.1", port)) == 0


def _find_available_port(start: int = 8515, end: int = 8530) -> int:
	for port in range(start, end + 1):
		if not _is_port_open(port):
			return port
	# Fallback if all checked ports are occupied.
	return end + 1


def ensure_predict_app_running() -> str:
	preferred_port = int(st.session_state.get("predict_app_port", 8515))
	port = preferred_port if _is_port_open(preferred_port) else _find_available_port(8515, 8530)

	if not _is_port_open(port):
		script_path = _predict_app_path()
		if not script_path.exists():
			raise FileNotFoundError(f"Prediction app script not found: {script_path}")

		command = [
			sys.executable,
			"-m",
			"streamlit",
			"run",
			str(script_path),
			"--server.headless",
			"true",
			"--server.address",
			"0.0.0.0",
			"--server.port",
			str(port),
		]
		subprocess.Popen(  # noqa: S603
			command,
			cwd=str(_project_root()),
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
		)
		# Give Streamlit a short window to initialize before redirect.
		for _ in range(12):
			if _is_port_open(port):
				break
			time.sleep(0.5)
		if not _is_port_open(port):
			raise RuntimeError(
				f"Prediction app did not start on port {port}. If running in Docker, publish this port using '-p {port}:{port}'."
			)

	st.session_state["predict_app_port"] = port
	dashboard_url = "http://localhost:8501"
	return f"http://localhost:{port}/?dashboard_url={quote_plus(dashboard_url)}"


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
	return subprocess.run(command, check=False, capture_output=True, text=True)


def _extract_batch_id(stdout_text: str) -> str | None:
	for line in stdout_text.splitlines():
		line = line.strip()
		if line.startswith("Created batch_id="):
			return line.split("=", maxsplit=1)[-1].strip()
	return None


def _risk_band(probability: float) -> str:
	if probability >= 0.8:
		return "High"
	if probability >= 0.65:
		return "Moderate"
	return "Elevated"


def _genai_suggestion_for_row(customer_identifier: str, probability: float) -> str:
	band = _risk_band(probability)
	if band == "High":
		return (
			f"Customer {customer_identifier} is high churn-risk. Trigger immediate save action: "
			"priority support call, 20% personalized discount, and 7-day cashback booster."
		)
	if band == "Moderate":
		return (
			f"Customer {customer_identifier} is moderate churn-risk. Send tailored retention journey: "
			"feature reminder, limited-time coupon, and one-click re-order recommendation."
		)
	return (
		f"Customer {customer_identifier} has elevated churn-risk. Use light-touch re-engagement: "
		"personalized app notification, loyalty points nudge, and category-based offer."
	)


def _build_genai_suggestions(predictions: pd.DataFrame) -> pd.DataFrame:
	churned = predictions[predictions["predicted_churn_flag"] == 1].copy()
	if churned.empty:
		return pd.DataFrame(columns=["customer_ref", "predicted_churn_probability", "risk_band", "genai_suggestion"])

	if "customer_id" in churned.columns:
		churned["customer_ref"] = churned["customer_id"].astype(str)
	else:
		churned["customer_ref"] = churned["row_number"].astype(str)

	churned["risk_band"] = churned["predicted_churn_probability"].apply(_risk_band)
	churned["genai_suggestion"] = churned.apply(
		lambda row: _genai_suggestion_for_row(
			customer_identifier=row["customer_ref"],
			probability=float(row["predicted_churn_probability"]),
		),
		axis=1,
	)

	return churned[["customer_ref", "predicted_churn_probability", "risk_band", "genai_suggestion"]]


def run_automated_batch_pipeline(sample_size: int = 100) -> dict[str, object]:
	root = _project_root()
	create_script = root / "create_batch.py"
	watcher_script = root / "watcher.py"
	pred_csv = root / "data" / "gold" / "latest_batch_predictions.csv"

	create_cmd = [sys.executable, str(create_script), "--sample-size", str(sample_size)]
	create_run = _run_command(create_cmd)
	if create_run.returncode != 0:
		raise RuntimeError(f"Batch creation failed: {create_run.stderr.strip() or create_run.stdout.strip()}")

	batch_id = _extract_batch_id(create_run.stdout)

	watcher_cmd = [sys.executable, str(watcher_script), "--source-mode", "both"]
	watcher_run = _run_command(watcher_cmd)
	if watcher_run.returncode != 0:
		raise RuntimeError(f"Watcher failed: {watcher_run.stderr.strip() or watcher_run.stdout.strip()}")

	if not pred_csv.exists():
		raise FileNotFoundError(f"Prediction output not found: {pred_csv}")

	predictions = pd.read_csv(pred_csv)
	if "predicted_churn_flag" not in predictions.columns:
		raise ValueError("Prediction file missing column: predicted_churn_flag")

	if batch_id and "batch_id" in predictions.columns:
		batch_predictions = predictions[predictions["batch_id"].astype(str) == str(batch_id)].copy()
		if batch_predictions.empty:
			batch_predictions = predictions.copy()
	else:
		batch_predictions = predictions.copy()

	batch_predictions["predicted_status"] = batch_predictions["predicted_churn_flag"].map(
		{1: "Churned", 0: "Retained"}
	).fillna("Unknown")

	suggestions = _build_genai_suggestions(batch_predictions)

	return {
		"batch_id": batch_id or "latest_batch",
		"predictions": batch_predictions,
		"suggestions": suggestions,
		"create_stdout": create_run.stdout,
		"watcher_stdout": watcher_run.stdout,
	}


def render_batch_prediction_outcome(result: dict[str, object]) -> None:
	predictions = result["predictions"]
	suggestions = result["suggestions"]
	batch_id = result["batch_id"]

	st.markdown("<div class='section-title'>ML Model Outcome (Automated ETL Batch)</div>", unsafe_allow_html=True)
	st.caption(f"Batch processed: {batch_id}")

	total_rows = int(len(predictions))
	churned_rows = int((predictions["predicted_churn_flag"] == 1).sum())
	retained_rows = int((predictions["predicted_churn_flag"] == 0).sum())

	m1, m2, m3 = st.columns(3)
	with m1:
		metric_card("Batch Predictions", f"{total_rows}")
	with m2:
		metric_card("Predicted Churned", f"{churned_rows}")
	with m3:
		metric_card("Predicted Retained", f"{retained_rows}")

	display_cols = [
		col
		for col in [
			"batch_id",
			"row_number",
			"customer_id",
			"predicted_churn_probability",
			"predicted_status",
		]
		if col in predictions.columns
	]

	st.dataframe(predictions[display_cols], use_container_width=True)

	if churned_rows > 0:
		st.markdown("<div class='section-title'>GenAI Suggestions for Churn-Risk Customers</div>", unsafe_allow_html=True)
		st.caption("Generated automatically when churn-risk customers are detected.")
		st.dataframe(suggestions, use_container_width=True)
	else:
		st.success("No churned customers predicted in this batch. GenAI suggestions were not required.")


st.markdown(
	f"""
	<style>
	@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');

	.stApp {{
		background:
			radial-gradient(circle at 5% 5%, #e6f4f1 0%, transparent 35%),
			radial-gradient(circle at 95% 8%, #ffe8d1 0%, transparent 30%),
			{THEME['bg']};
		color: {THEME['ink']};
	}}

	h1, h2, h3 {{
		font-family: 'Space Grotesk', sans-serif;
		letter-spacing: 0.2px;
	}}

	p, span, li, div {{
		font-family: 'Manrope', sans-serif;
	}}

	.hero-wrap {{
		background: linear-gradient(120deg, #0f766e 0%, #115e59 55%, #134e4a 100%);
		border-radius: 18px;
		padding: 1.4rem 1.6rem;
		color: #f8fafc;
		border: 1px solid rgba(255,255,255,0.16);
		box-shadow: 0 12px 28px rgba(17, 94, 89, 0.24);
		margin-bottom: 0.8rem;
	}}

	.hero-kicker {{
		opacity: 0.88;
		text-transform: uppercase;
		font-size: 0.76rem;
		letter-spacing: 1.1px;
	}}

	.hero-title {{
		font-size: 1.95rem;
		margin: 0.3rem 0 0.2rem 0;
		font-family: 'Space Grotesk', sans-serif;
		font-weight: 700;
	}}

	.hero-sub {{
		font-size: 0.98rem;
		margin: 0;
		opacity: 0.96;
	}}

	.metric-card {{
		background: {THEME['card']};
		border: 1px solid {THEME['line']};
		border-radius: 14px;
		padding: 0.85rem 1rem;
		box-shadow: 0 7px 18px rgba(0, 0, 0, 0.05);
		min-height: 98px;
	}}

	.metric-label {{
		font-size: 0.82rem;
		color: {THEME['muted']};
		margin-bottom: 0.25rem;
		text-transform: uppercase;
		letter-spacing: 0.6px;
	}}

	.metric-value {{
		font-size: 1.5rem;
		font-weight: 700;
		font-family: 'Space Grotesk', sans-serif;
	}}

	.section-title {{
		font-family: 'Space Grotesk', sans-serif;
		font-size: 1.18rem;
		margin: 0.4rem 0 0.55rem 0;
	}}

	.note {{
		color: {THEME['muted']};
		font-size: 0.9rem;
	}}
	</style>
	""",
	unsafe_allow_html=True,
)


def _candidate_data_paths() -> list[Path]:
	root = Path(__file__).resolve().parents[2]
	return [
		root / "data" / "preprocessed" / "churn_data_preprocessed.csv",
		root / "data" / "preprocessed" / "churn_data.csv",
	]


def _warehouse_db_path() -> Path:
	root = Path(__file__).resolve().parents[2]
	return root / "data" / "gold" / "warehouse.db"


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
	db_path = _warehouse_db_path()
	if db_path.exists():
		query = """
		SELECT
			dcp.customer_id AS CustomerID,
			fcm.churn_flag AS Churn,
			COALESCE(fcm.customer_tenure_months, dctw.customer_tenure_months) AS Tenure,
			dld.preferred_login_device AS PreferredLoginDevice,
			dcp.city_tier AS CityTier,
			fcm.warehouse_to_home_distance AS WarehouseToHome,
			dpm.preferred_payment_mode AS PreferredPaymentMode,
			dcp.customer_gender AS Gender,
			fcm.hours_spent_on_app AS HourSpendOnApp,
			fcm.registered_device_count AS NumberOfDeviceRegistered,
			doc.preferred_order_category AS PreferedOrderCat,
			fcm.satisfaction_score AS SatisfactionScore,
			dcp.marital_status AS MaritalStatus,
			fcm.address_count AS NumberOfAddress,
			fcm.complaint_flag AS Complain,
			fcm.order_amount_hike_from_last_year AS OrderAmountHikeFromlastYear,
			fcm.coupon_used_count AS CouponUsed,
			fcm.order_count AS OrderCount,
			COALESCE(fcm.days_since_last_order, dctw.days_since_last_order) AS DaySinceLastOrder,
			fcm.cashback_amount AS CashbackAmount
		FROM fact_customer_churn_metrics fcm
		LEFT JOIN dim_customer_profile dcp
			ON dcp.customer_profile_key = fcm.customer_profile_key
		LEFT JOIN dim_customer_time_window dctw
			ON dctw.customer_time_key = fcm.customer_time_key
		LEFT JOIN dim_login_device dld
			ON dld.login_device_key = fcm.login_device_key
		LEFT JOIN dim_payment_method dpm
			ON dpm.payment_method_key = fcm.payment_method_key
		LEFT JOIN dim_order_category doc
			ON doc.order_category_key = fcm.order_category_key
		"""
		with sqlite3.connect(db_path) as conn:
			df = pd.read_sql_query(query, conn)
		if not df.empty:
			return df, f"warehouse.db ({db_path})"

	for path in _candidate_data_paths():
		if path.exists():
			return pd.read_csv(path), f"CSV fallback ({path})"

	raise FileNotFoundError(
		"Could not load preprocessed dataset from warehouse.db or CSV fallback."
	)


def normalize_types(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	numeric_candidates = [
		"Churn",
		"Tenure",
		"CityTier",
		"WarehouseToHome",
		"HourSpendOnApp",
		"NumberOfDeviceRegistered",
		"SatisfactionScore",
		"NumberOfAddress",
		"Complain",
		"OrderAmountHikeFromlastYear",
		"CouponUsed",
		"OrderCount",
		"DaySinceLastOrder",
		"CashbackAmount",
	]
	for col in numeric_candidates:
		if col in out.columns:
			out[col] = pd.to_numeric(out[col], errors="coerce")

	if "Churn" in out.columns:
		out["ChurnLabel"] = out["Churn"].map({1: "Churn", 0: "Retained"}).fillna("Unknown")

	return out


def quality_summary(df: pd.DataFrame) -> dict[str, float]:
	total_cells = df.shape[0] * df.shape[1]
	missing_cells = int(df.isna().sum().sum())
	duplicates = int(df.duplicated().sum())
	completeness = (1 - (missing_cells / total_cells)) * 100 if total_cells else 0
	return {
		"total_rows": int(df.shape[0]),
		"total_cols": int(df.shape[1]),
		"missing_cells": missing_cells,
		"duplicate_rows": duplicates,
		"completeness": round(completeness, 2),
		"missing_pct": round((missing_cells / total_cells) * 100, 2) if total_cells else 0,
	}


def outlier_counts_iqr(df: pd.DataFrame) -> pd.DataFrame:
	numeric_df = df.select_dtypes(include=[np.number])
	records: list[dict[str, float]] = []
	for col in numeric_df.columns:
		series = numeric_df[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		if iqr == 0:
			count = 0
		else:
			lower = q1 - 1.5 * iqr
			upper = q3 + 1.5 * iqr
			count = int(((series < lower) | (series > upper)).sum())
		records.append({"feature": col, "outliers": count})

	out = pd.DataFrame(records)
	if out.empty:
		return out
	return out.sort_values("outliers", ascending=False).head(12)


def metric_card(label: str, value: str) -> None:
	st.markdown(
		f"""
		<div class="metric-card">
			<div class="metric-label">{label}</div>
			<div class="metric-value">{value}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def styled_plot(fig: go.Figure) -> go.Figure:
	fig.update_layout(
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(255,255,255,0.75)",
		font=dict(family="Manrope, sans-serif", color=THEME["ink"]),
		title_font=dict(family="Space Grotesk, sans-serif", size=20),
		margin=dict(l=20, r=20, t=55, b=20),
		legend_title_text="",
	)
	return fig


def render_global_overview(df: pd.DataFrame) -> None:
	qs = quality_summary(df)
	churn_rate = (df["Churn"].mean() * 100) if "Churn" in df.columns else 0.0

	c1, c2, c3, c4 = st.columns(4)
	with c1:
		metric_card("Total Customers", f"{qs['total_rows']:,}")
	with c2:
		metric_card("Churn Rate", f"{churn_rate:.2f}%")
	with c3:
		metric_card("Data Completeness", f"{qs['completeness']:.2f}%")
	with c4:
		metric_card("Duplicate Rows", f"{qs['duplicate_rows']:,}")

	st.markdown("<div class='section-title'>Global EDA View</div>", unsafe_allow_html=True)
	left, right = st.columns((1.1, 1.6))

	with left:
		if "ChurnLabel" in df.columns:
			churn_pie = px.pie(
				df,
				names="ChurnLabel",
				color="ChurnLabel",
				hole=0.58,
				color_discrete_map={
					"Retained": "#0f766e",
					"Churn": "#dc2626",
					"Unknown": "#9ca3af",
				},
				title="Customer Retention Mix",
			)
			churn_pie.update_traces(textposition="inside", textinfo="percent+label")
			st.plotly_chart(styled_plot(churn_pie), use_container_width=True)

		if "Tenure" in df.columns:
			tenure_box = px.box(
				df,
				x="ChurnLabel" if "ChurnLabel" in df.columns else None,
				y="Tenure",
				color="ChurnLabel" if "ChurnLabel" in df.columns else None,
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626"},
				title="Tenure Spread by Outcome",
				points=False,
			)
			st.plotly_chart(styled_plot(tenure_box), use_container_width=True)

	with right:
		if {"PreferredPaymentMode", "Churn"}.issubset(df.columns):
			pay_mode = (
				df.groupby("PreferredPaymentMode", dropna=False)["Churn"]
				.mean()
				.sort_values(ascending=False)
				.reset_index()
			)
			pay_mode["ChurnRate"] = pay_mode["Churn"] * 100
			pay_fig = px.bar(
				pay_mode,
				x="PreferredPaymentMode",
				y="ChurnRate",
				color="ChurnRate",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
				title="Churn Rate by Payment Mode",
			)
			pay_fig.update_yaxes(title="Churn Rate (%)")
			pay_fig.update_xaxes(title="")
			st.plotly_chart(styled_plot(pay_fig), use_container_width=True)

		if {"CityTier", "Churn"}.issubset(df.columns):
			city_heat = (
				df.groupby(["CityTier", "Churn"], dropna=False)
				.size()
				.reset_index(name="Customers")
			)
			city_heat["ChurnLabel"] = city_heat["Churn"].map({0: "Retained", 1: "Churn"})
			heat = px.density_heatmap(
				city_heat,
				x="CityTier",
				y="ChurnLabel",
				z="Customers",
				color_continuous_scale="YlOrRd",
				title="City Tier vs Churn Density",
			)
			st.plotly_chart(styled_plot(heat), use_container_width=True)


# def render_data_quality(df: pd.DataFrame) -> None:
# 	st.markdown("<div class='section-title'>Data Quality Control Tower</div>", unsafe_allow_html=True)
# 	qs = quality_summary(df)

# 	l1, l2 = st.columns((1.0, 1.8))
# 	with l1:
# 		gauge = go.Figure(
# 			go.Indicator(
# 				mode="gauge+number",
# 				value=qs["completeness"],
# 				title={"text": "Completeness %"},
# 				gauge={
# 					"axis": {"range": [0, 100]},
# 					"bar": {"color": "#0f766e"},
# 					"steps": [
# 						{"range": [0, 80], "color": "#fee2e2"},
# 						{"range": [80, 95], "color": "#fef3c7"},
# 						{"range": [95, 100], "color": "#dcfce7"},
# 					],
# 				},
# 			)
# 		)
# 		st.plotly_chart(styled_plot(gauge), use_container_width=True)

# 	with l2:
# 		st.markdown("<div class='section-title'>Quality Snapshot</div>", unsafe_allow_html=True)
# 		q1, q2, q3 = st.columns(3)
# 		with q1:
# 			metric_card("Missing Cells", f"{qs['missing_cells']:,}")
# 		with q2:
# 			metric_card("Missing %", f"{qs['missing_pct']:.2f}%")
# 		with q3:
# 			metric_card("Columns", f"{qs['total_cols']:,}")


def marketing_analytics_view(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Marketing Analytics Persona</div>", unsafe_allow_html=True)
	st.caption("Campaign targeting, retention messaging, and segment conversion signals.")

	# 1) Churn Rate Analysis
	r1c1, r1c2 = st.columns(2)
	with r1c1:
		if "ChurnLabel" in df.columns:
			churn_mix = px.pie(
				df,
				names="ChurnLabel",
				color="ChurnLabel",
				hole=0.6,
				title="Churn Rate Analysis",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626", "Unknown": "#9ca3af"},
			)
			churn_mix.update_traces(textposition="inside", textinfo="percent+label")
			st.plotly_chart(styled_plot(churn_mix), use_container_width=True)

	# 2) City Tier Market Segmentation
	with r1c2:
		if {"CityTier", "OrderCount"}.issubset(df.columns):
			city_orders = (
				df.groupby("CityTier", dropna=False)["OrderCount"]
				.sum()
				.reset_index(name="TotalOrders")
				.sort_values("TotalOrders", ascending=False)
			)
			city_orders["CityName"] = city_orders["CityTier"].map(CITY_TIER_NAME_MAP).fillna(
				"Unknown City"
			)
			city_fig = px.bar(
				city_orders,
				x="CityName",
				y="TotalOrders",
				color="TotalOrders",
				title="City Tier Market Segmentation (Orders)",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
			)
			city_fig.update_xaxes(title="City")
			city_fig.update_yaxes(title="Total Orders")
			st.plotly_chart(styled_plot(city_fig), use_container_width=True)

	# 3) Product Category Popularity
	r2c1, r2c2 = st.columns(2)
	with r2c1:
		if "PreferedOrderCat" in df.columns:
			cat_popularity = (
				df["PreferedOrderCat"]
				.fillna("Unknown")
				.value_counts()
				.reset_index()
			)
			cat_popularity.columns = ["PreferedOrderCat", "Customers"]
			cat_fig = px.bar(
				cat_popularity,
				x="PreferedOrderCat",
				y="Customers",
				color="Customers",
				title="Product Category Popularity",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
			)
			cat_fig.update_xaxes(tickangle=-20, title="Preferred Order Category")
			st.plotly_chart(styled_plot(cat_fig), use_container_width=True)

	# 4) Coupon Effectiveness
	with r2c2:
		if {"CouponUsed", "OrderCount"}.issubset(df.columns):
			coupon_effect = (
				df.groupby("CouponUsed", dropna=False)["OrderCount"]
				.mean()
				.reset_index(name="AvgOrderCount")
				.sort_values("CouponUsed")
			)
			coupon_effect["CouponUsed"] = coupon_effect["CouponUsed"].fillna(0)
			coupon_fig = px.bar(
				coupon_effect,
				x="CouponUsed",
				y="AvgOrderCount",
				color="AvgOrderCount",
				title="Coupon Effectiveness (Avg Orders by Coupon Usage)",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
			)
			coupon_fig.update_xaxes(title="Coupons Used")
			coupon_fig.update_yaxes(title="Average Order Count")
			st.plotly_chart(styled_plot(coupon_fig), use_container_width=True)

	# 5) Cashback Marketing Impact
	r3c1, r3c2 = st.columns(2)
	with r3c1:
		if {"CashbackAmount", "OrderCount", "ChurnLabel"}.issubset(df.columns):
			cashback_fig = px.scatter(
				df,
				x="CashbackAmount",
				y="OrderCount",
				color="ChurnLabel",
				opacity=0.66,
				title="Cashback Marketing Impact",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626"},
			)
			cashback_fig.update_xaxes(title="Cashback Amount")
			cashback_fig.update_yaxes(title="Order Count")
			st.plotly_chart(styled_plot(cashback_fig), use_container_width=True)

	# 6) Customer Loyalty Growth
	with r3c2:
		if "OrderAmountHikeFromlastYear" in df.columns:
			growth_fig = px.histogram(
				df,
				x="OrderAmountHikeFromlastYear",
				nbins=30,
				color_discrete_sequence=["#0f766e"],
				title="Customer Loyalty Growth",
			)
			growth_fig.update_xaxes(title="Order Amount Hike From Last Year")
			growth_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(growth_fig), use_container_width=True)

	# 7) Purchase Frequency Analysis
	if "OrderCount" in df.columns:
		freq_fig = px.histogram(
			df,
			x="OrderCount",
			nbins=25,
			color_discrete_sequence=["#d97706"],
			title="Purchase Frequency Analysis",
		)
		freq_fig.update_xaxes(title="Order Count")
		freq_fig.update_yaxes(title="Customer Count")
		st.plotly_chart(styled_plot(freq_fig), use_container_width=True)


def product_manager_view(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Product Manager Persona</div>", unsafe_allow_html=True)
	st.caption("Product Engagement Analytics: app usage, device behavior, recency, and churn linkage.")

	# 1) App Usage Analysis
	r1c1, r1c2 = st.columns(2)
	with r1c1:
		if "HourSpendOnApp" in df.columns:
			usage_fig = px.histogram(
				df,
				x="HourSpendOnApp",
				nbins=30,
				color_discrete_sequence=["#0f766e"],
				title="App Usage Analysis",
			)
			usage_fig.update_xaxes(title="Hours Spent on App")
			usage_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(usage_fig), use_container_width=True)

	# 2) Device Preference Analysis
	with r1c2:
		if "PreferredLoginDevice" in df.columns:
			device_pref = (
				df["PreferredLoginDevice"].fillna("Unknown").value_counts().reset_index()
			)
			device_pref.columns = ["PreferredLoginDevice", "Customers"]
			device_fig = px.bar(
				device_pref,
				x="PreferredLoginDevice",
				y="Customers",
				color="Customers",
				title="Device Preference Analysis",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
			)
			st.plotly_chart(styled_plot(device_fig), use_container_width=True)

	# 3) Multi-Device Engagement
	r2c1, r2c2 = st.columns(2)
	with r2c1:
		if "NumberOfDeviceRegistered" in df.columns:
			multi_device_fig = px.histogram(
				df,
				x="NumberOfDeviceRegistered",
				nbins=15,
				color_discrete_sequence=["#d97706"],
				title="Multi-Device Engagement",
			)
			multi_device_fig.update_xaxes(title="Number of Devices Registered")
			multi_device_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(multi_device_fig), use_container_width=True)

	# 4) Recency Analysis
	with r2c2:
		if "DaySinceLastOrder" in df.columns:
			recency_fig = px.histogram(
				df,
				x="DaySinceLastOrder",
				nbins=30,
				color_discrete_sequence=["#ef4444"],
				title="Recency Analysis",
			)
			recency_fig.update_xaxes(title="Days Since Last Order")
			recency_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(recency_fig), use_container_width=True)

	# 5) Engagement vs Purchases
	r3c1, r3c2 = st.columns(2)
	with r3c1:
		if {"HourSpendOnApp", "OrderCount", "ChurnLabel"}.issubset(df.columns):
			engagement_purchase_fig = px.scatter(
				df,
				x="HourSpendOnApp",
				y="OrderCount",
				color="ChurnLabel",
				opacity=0.64,
				title="Engagement vs Purchases",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626"},
			)
			st.plotly_chart(styled_plot(engagement_purchase_fig), use_container_width=True)

	# 6) Device Usage vs Engagement
	with r3c2:
		if {"PreferredLoginDevice", "HourSpendOnApp"}.issubset(df.columns):
			device_engagement_fig = px.box(
				df,
				x="PreferredLoginDevice",
				y="HourSpendOnApp",
				color="PreferredLoginDevice",
				title="Device Usage vs Engagement",
			)
			device_engagement_fig.update_xaxes(title="Preferred Login Device")
			device_engagement_fig.update_yaxes(title="Hours Spent on App")
			st.plotly_chart(styled_plot(device_engagement_fig), use_container_width=True)

	# 7) Engagement vs Churn
	if {"HourSpendOnApp", "ChurnLabel"}.issubset(df.columns):
		engagement_churn_fig = px.box(
			df,
			x="ChurnLabel",
			y="HourSpendOnApp",
			color="ChurnLabel",
			title="Engagement vs Churn",
			color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626", "Unknown": "#9ca3af"},
		)
		engagement_churn_fig.update_xaxes(title="Churn Status")
		engagement_churn_fig.update_yaxes(title="Hours Spent on App")
		st.plotly_chart(styled_plot(engagement_churn_fig), use_container_width=True)


def support_customer_view(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Support Customer Persona</div>", unsafe_allow_html=True)
	st.caption("Customer Support Analytics: service quality, complaints, and delivery experience impact.")

	# 1) Customer Satisfaction Distribution
	r1c1, r1c2 = st.columns(2)
	with r1c1:
		if "SatisfactionScore" in df.columns:
			sat_fig = px.histogram(
				df,
				x="SatisfactionScore",
				nbins=10,
				color_discrete_sequence=["#0f766e"],
				title="Customer Satisfaction Distribution",
			)
			sat_fig.update_xaxes(title="Satisfaction Score")
			sat_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(sat_fig), use_container_width=True)

	# 2) Complaints Analysis
	with r1c2:
		if "Complain" in df.columns:
			complain_map = df["Complain"].map({0: "No Complaint", 1: "Complaint"}).fillna("Unknown")
			complaints_fig = px.pie(
				complain_map.to_frame(name="ComplainLabel"),
				names="ComplainLabel",
				hole=0.55,
				title="Complaints Analysis",
				color="ComplainLabel",
				color_discrete_map={"No Complaint": "#0f766e", "Complaint": "#dc2626", "Unknown": "#9ca3af"},
			)
			complaints_fig.update_traces(textposition="inside", textinfo="percent+label")
			st.plotly_chart(styled_plot(complaints_fig), use_container_width=True)

	# 3) Complaints vs Churn
	r2c1, r2c2 = st.columns(2)
	with r2c1:
		if {"Complain", "Churn"}.issubset(df.columns):
			comp_churn = (
				df.assign(
					ComplainLabel=df["Complain"].map({0: "No Complaint", 1: "Complaint"}).fillna("Unknown"),
					ChurnLabel=df["Churn"].map({0: "Retained", 1: "Churn"}).fillna("Unknown"),
				)
				.groupby(["ComplainLabel", "ChurnLabel"], dropna=False)
				.size()
				.reset_index(name="Customers")
			)
			comp_churn_fig = px.bar(
				comp_churn,
				x="ComplainLabel",
				y="Customers",
				color="ChurnLabel",
				barmode="group",
				title="Complaints vs Churn",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626", "Unknown": "#9ca3af"},
			)
			st.plotly_chart(styled_plot(comp_churn_fig), use_container_width=True)

	# 4) Delivery Distance Impact
	with r2c2:
		if "WarehouseToHome" in df.columns:
			distance_hist = px.histogram(
				df,
				x="WarehouseToHome",
				nbins=30,
				color_discrete_sequence=["#d97706"],
				title="Delivery Distance Impact",
			)
			distance_hist.update_xaxes(title="Warehouse to Home Distance")
			distance_hist.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(distance_hist), use_container_width=True)

	# 5) Distance vs Satisfaction
	r3c1, r3c2 = st.columns(2)
	with r3c1:
		if {"WarehouseToHome", "SatisfactionScore", "ChurnLabel"}.issubset(df.columns):
			distance_sat = px.scatter(
				df,
				x="WarehouseToHome",
				y="SatisfactionScore",
				color="ChurnLabel",
				opacity=0.62,
				title="Distance vs Satisfaction",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626", "Unknown": "#9ca3af"},
			)
			st.plotly_chart(styled_plot(distance_sat), use_container_width=True)

	# 6) Payment Mode Analysis
	with r3c2:
		if "PreferredPaymentMode" in df.columns:
			pay_mode = df["PreferredPaymentMode"].fillna("Unknown").value_counts().reset_index()
			pay_mode.columns = ["PreferredPaymentMode", "Customers"]
			pay_mode_fig = px.bar(
				pay_mode,
				x="PreferredPaymentMode",
				y="Customers",
				color="Customers",
				title="Payment Mode Analysis",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
			)
			pay_mode_fig.update_xaxes(tickangle=-20, title="Preferred Payment Mode")
			st.plotly_chart(styled_plot(pay_mode_fig), use_container_width=True)

	# 7) Address Complexity Analysis
	if "NumberOfAddress" in df.columns:
		address_dist = (
			df["NumberOfAddress"].dropna().astype(int).value_counts().sort_index().reset_index()
		)
		address_dist.columns = ["NumberOfAddress", "CustomerCount"]
		address_fig = px.bar(
			address_dist,
			x="NumberOfAddress",
			y="CustomerCount",
			color="CustomerCount",
			title="Address Complexity Analysis",
			color_continuous_scale=["#dbeafe", "#60a5fa", "#1d4ed8"],
		)
		address_fig.update_xaxes(title="Number of Addresses")
		address_fig.update_yaxes(title="Customer Count")
		st.plotly_chart(styled_plot(address_fig), use_container_width=True)


def strategy_persona_view(df: pd.DataFrame) -> None:
	st.markdown("<div class='section-title'>Strategy Persona</div>", unsafe_allow_html=True)
	st.caption("Executive-level market picture for strategic retention planning.")
	st.markdown("<div class='section-title'>Executive Overview Dashboard</div>", unsafe_allow_html=True)

	# 1) Total Customers KPI
	total_customers = int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else int(df.shape[0])
	# 2) Churn Rate KPI
	churn_rate = float(df["Churn"].mean() * 100) if "Churn" in df.columns else 0.0
	# 3) Average Customer Tenure
	avg_tenure = float(df["Tenure"].mean()) if "Tenure" in df.columns else 0.0
	# 4) Total Orders Overview
	total_orders = float(df["OrderCount"].sum()) if "OrderCount" in df.columns else 0.0

	k1, k2, k3, k4 = st.columns(4)
	with k1:
		metric_card("Total Customers", f"{total_customers:,}")
	with k2:
		metric_card("Churn Rate", f"{churn_rate:.2f}%")
	with k3:
		metric_card("Average Tenure", f"{avg_tenure:.2f}")
	with k4:
		metric_card("Total Orders", f"{total_orders:,.0f}")

	r1c1, r1c2 = st.columns(2)
	with r1c1:
		if "ChurnLabel" in df.columns:
			churn_pie = px.pie(
				df,
				names="ChurnLabel",
				color="ChurnLabel",
				hole=0.58,
				title="Churn Rate KPI",
				color_discrete_map={"Retained": "#0f766e", "Churn": "#dc2626", "Unknown": "#9ca3af"},
			)
			churn_pie.update_traces(textposition="inside", textinfo="percent+label")
			st.plotly_chart(styled_plot(churn_pie), use_container_width=True)

	with r1c2:
		if "SatisfactionScore" in df.columns:
			avg_satisfaction = float(df["SatisfactionScore"].mean())
			sat_gauge = go.Figure(
				go.Indicator(
					mode="gauge+number",
					value=avg_satisfaction,
					title={"text": "Customer Satisfaction Score"},
					gauge={
						"axis": {"range": [0, 5]},
						"bar": {"color": "#0f766e"},
						"steps": [
							{"range": [0, 2], "color": "#fee2e2"},
							{"range": [2, 3.5], "color": "#fef3c7"},
							{"range": [3.5, 5], "color": "#dcfce7"},
						],
					},
				)
			)
			st.plotly_chart(styled_plot(sat_gauge), use_container_width=True)

	r2c1, r2c2 = st.columns(2)
	with r2c1:
		if "OrderAmountHikeFromlastYear" in df.columns:
			growth_fig = px.histogram(
				df,
				x="OrderAmountHikeFromlastYear",
				nbins=30,
				color_discrete_sequence=["#d97706"],
				title="Revenue Growth Indicator",
			)
			growth_fig.update_xaxes(title="Order Amount Hike From Last Year")
			growth_fig.update_yaxes(title="Customer Count")
			st.plotly_chart(styled_plot(growth_fig), use_container_width=True)

	with r2c2:
		if {"HourSpendOnApp", "OrderCount", "DaySinceLastOrder"}.issubset(df.columns):
			activity_fig = px.scatter(
				df,
				x="HourSpendOnApp",
				y="OrderCount",
				color="DaySinceLastOrder",
				opacity=0.65,
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
				title="Customer Activity Summary",
			)
			activity_fig.update_xaxes(title="Hour Spend On App")
			activity_fig.update_yaxes(title="Order Count")
			st.plotly_chart(styled_plot(activity_fig), use_container_width=True)

	c1, c2 = st.columns(2)
	with c1:
		if {"CityTier", "MaritalStatus", "Churn"}.issubset(df.columns):
			strat = (
				df.groupby(["CityTier", "MaritalStatus"], dropna=False)["Churn"]
				.mean()
				.mul(100)
				.reset_index(name="ChurnRate")
			)
			strat["CityName"] = strat["CityTier"].map(CITY_TIER_NAME_MAP).fillna("Unknown City")
			fig = px.treemap(
				strat,
				path=["CityName", "MaritalStatus"],
				values="ChurnRate",
				color="ChurnRate",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
				title="Strategic Churn Heat by City and Marital Segment",
			)
			st.plotly_chart(styled_plot(fig), use_container_width=True)

	with c2:
		if {"Tenure", "OrderCount", "CashbackAmount", "Churn"}.issubset(df.columns):
			summary = pd.DataFrame(
				{
					"Metric": ["Avg Tenure", "Avg Orders", "Avg Cashback", "Churn Rate"],
					"Value": [
						df["Tenure"].mean(),
						df["OrderCount"].mean(),
						df["CashbackAmount"].mean(),
						df["Churn"].mean() * 100,
					],
				}
			)
			fig = px.bar_polar(
				summary,
				r="Value",
				theta="Metric",
				color="Value",
				color_continuous_scale=["#0f766e", "#f59e0b", "#dc2626"],
				title="Strategic KPI Radar",
			)
			st.plotly_chart(styled_plot(fig), use_container_width=True)


def main() -> None:
	st.markdown(
		"""
		<div class="hero-wrap">
			<div class="hero-kicker">Naive Bar Persona Dashboard</div>
			<div class="hero-title">Preprocessed Churn Data Intelligence</div>
			<p class="hero-sub">Global EDA and Data Quality Storytelling for Marketing Analytics, Product Manager, Support Customer, and Strategy Persona.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	try:
		df_raw, data_source = load_data()
	except Exception as exc:
		st.error(f"Failed to load preprocessed data: {exc}")
		st.stop()

	df = normalize_types(df_raw)

	persona_options = [
		"1. Marketing Analytics",
		"2. Product Manager",
		"3. Support Customer",
		"4. Strategy Persona",
	]
	persona = st.sidebar.selectbox(
		"Naive Bar",
		persona_options,
		index=0,
		key="naive_bar_persona",
		help="Choose a persona to update the dashboard view.",
	)

	st.sidebar.markdown("---")
	st.sidebar.markdown("### Individual Prediction")
	st.sidebar.caption("Open single-customer churn prediction with GenAI message output.")
	if st.sidebar.button("GenAI Retention Assistant (Individual)", use_container_width=True):
		try:
			predict_url = ensure_predict_app_running()
			st.session_state["predict_redirect_url"] = predict_url
			st.session_state["predict_do_redirect"] = True
			st.sidebar.success("Prediction app ready. Redirecting...")
		except Exception as exc:  # noqa: BLE001
			st.sidebar.error(f"Unable to open individual predictor: {exc}")

	if "predict_redirect_url" in st.session_state:
		url = st.session_state["predict_redirect_url"]
		st.link_button("Open Individual Prediction Output", url, use_container_width=True)
		if st.session_state.get("predict_do_redirect", False):
			st.markdown(
				f"<meta http-equiv='refresh' content='0;url={url}'>",
				unsafe_allow_html=True,
			)
			st.session_state["predict_do_redirect"] = False

	st.sidebar.markdown("---")
	st.sidebar.markdown("### Batch Prediction Trigger")
	st.sidebar.caption("Create 100-customer batch, run automated ETL + ML, then show churn outcomes.")

	if st.sidebar.button("ML Batch Churn Prediction (100)", type="primary"):
		with st.spinner("Running automated ETL workflow and ML inference..."):
			try:
				result = run_automated_batch_pipeline(sample_size=100)
				st.session_state["latest_batch_result"] = result
				st.session_state["show_batch_only"] = True
				st.sidebar.success("Batch pipeline completed successfully.")
			except Exception as exc:  # noqa: BLE001
				st.sidebar.error(f"Pipeline failed: {exc}")

	if "latest_batch_result" in st.session_state:
		render_batch_prediction_outcome(st.session_state["latest_batch_result"])
		if st.button("Back to Persona Insights", use_container_width=True):
			st.session_state["show_batch_only"] = False
			st.session_state.pop("latest_batch_result", None)
			st.rerun()

	# When batch prediction output is shown, suppress lower persona/dashboard graphs.
	if st.session_state.get("show_batch_only", False):
		return

	with st.expander("Dataset Scope", expanded=False):
		st.write(f"Source: {data_source}")
		st.write(f"Rows: {df.shape[0]:,}")
		st.write(f"Columns: {df.shape[1]}")
		st.dataframe(df, use_container_width=True)

	# Keep 1) Marketing and 2) Product fully separated with only persona-specific visuals.
	if persona.startswith("1."):
		marketing_analytics_view(df)
	elif persona.startswith("2."):
		product_manager_view(df)
	elif persona.startswith("3."):
		render_global_overview(df)
		st.markdown("<div class='section-title'>Persona Command Lens</div>", unsafe_allow_html=True)
		support_customer_view(df)
	else:
		render_global_overview(df)
		st.markdown("<div class='section-title'>Persona Command Lens</div>", unsafe_allow_html=True)
		strategy_persona_view(df)

	st.markdown(
		"<p class='note'>Tip: Use Streamlit sidebar persona switch to rapidly compare the same data from four decision lenses.</p>",
		unsafe_allow_html=True,
	)


if __name__ == "__main__":
	main()

