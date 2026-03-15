"""Streamlit app for customer churn prediction + retention messaging.

This app:
1) Loads model artifacts from ml/churn_model.pkl.
2) Collects all customer input columns via a form.
3) Predicts churn label and probability.
4) Generates a personalized retention message using genai/retention_agent.py.
"""

from __future__ import annotations

import pickle
import sys
import socket
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ml" / "churn_model.pkl"
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "silver" / "cleaned_data.csv"

if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

RETENTION_IMPORT_ERROR: str | None = None

try:
	from genai.retention_agent import (  # noqa: E402
		DEFAULT_MODEL_ID,
		build_fallback_explanation,
		build_fallback_retention_message,
		create_hf_client,
		generate_customer_outputs,
		get_hf_api_token,
		identify_risk_signals,
		resolve_provider,
	)
except Exception as exc:  # noqa: BLE001
	RETENTION_IMPORT_ERROR = str(exc)
	DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

	def identify_risk_signals(customer: pd.Series) -> list[str]:
		signals: list[str] = []
		if float(customer.get("DaySinceLastOrder", 0)) >= 7:
			signals.append("High inactivity since last order")
		if float(customer.get("OrderCount", 0)) <= 1:
			signals.append("Very low recent order count")
		if float(customer.get("HourSpendOnApp", 0)) <= 2:
			signals.append("Low app engagement")
		if not signals:
			signals.append("General elevated churn-risk pattern from model output")
		return signals

	def build_fallback_explanation(risk_signals: list[str]) -> str:
		return (
			"The customer shows elevated churn risk based on engagement and purchase behavior, "
			f"including: {', '.join(risk_signals[:3])}."
		)

	def build_fallback_retention_message(customer: pd.Series, risk_signals: list[str]) -> str:
		customer_id = customer.get("CustomerID", "NA")
		primary_signal = risk_signals[0] if risk_signals else "recent engagement decline"
		return (
			f"Dear Customer {customer_id}, we value your relationship with us and noticed "
			f"{str(primary_signal).lower()}. As a goodwill gesture, we are offering you "
			"a personalized 15% discount and priority support on your next purchase. "
			"Please use code STAY15 within 7 days."
		)


CATEGORICAL_FALLBACKS: dict[str, list[str]] = {
	"PreferredLoginDevice": ["Mobile Phone", "Phone", "Computer"],
	"PreferredPaymentMode": [
		"Debit Card",
		"Credit Card",
		"UPI",
		"Cash on Delivery",
		"COD",
		"E wallet",
	],
	"Gender": ["Female", "Male"],
	"PreferedOrderCat": ["Mobile", "Laptop & Accessory", "Grocery", "Others", "Mobile Phone"],
	"MaritalStatus": ["Single", "Married", "Divorced"],
}


NUMERIC_FALLBACKS: dict[str, float] = {
	"CustomerID": 99999,
	"Tenure": 6,
	"CityTier": 1,
	"WarehouseToHome": 12,
	"HourSpendOnApp": 3,
	"NumberOfDeviceRegistered": 3,
	"SatisfactionScore": 3,
	"NumberOfAddress": 2,
	"Complain": 0,
	"OrderAmountHikeFromlastYear": 15,
	"CouponUsed": 1,
	"OrderCount": 1,
	"DaySinceLastOrder": 3,
	"CashbackAmount": 140,
}


def get_category_options(reference_df: pd.DataFrame, column_name: str) -> list[str]:
	"""Return category options from reference data or fallback."""
	fallback = CATEGORICAL_FALLBACKS[column_name]
	if reference_df.empty or column_name not in reference_df.columns:
		return fallback

	values = sorted({str(value) for value in reference_df[column_name].dropna().tolist()})
	if values:
		return values

	return fallback


def get_numeric_default(reference_df: pd.DataFrame, column_name: str) -> float:
	"""Return median-based numeric default from reference data or fallback."""
	fallback = float(NUMERIC_FALLBACKS[column_name])
	if reference_df.empty or column_name not in reference_df.columns:
		return fallback

	series = pd.to_numeric(reference_df[column_name], errors="coerce").dropna()
	if series.empty:
		return fallback

	return float(series.median())


@st.cache_resource
def load_model_artifacts() -> dict[str, Any]:
	"""Load serialized churn model bundle from pkl file."""
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

	with MODEL_PATH.open("rb") as file:
		artifacts = pickle.load(file)

	if not isinstance(artifacts, dict):
		raise ValueError("Expected model bundle as dict in churn_model.pkl")

	required_keys = {"model", "scaler", "feature_columns"}
	missing = required_keys - set(artifacts.keys())
	if missing:
		raise ValueError(f"Model bundle missing keys: {sorted(missing)}")

	return artifacts


@st.cache_data
def load_reference_data() -> pd.DataFrame:
	"""Load reference churn dataset for better form defaults/options."""
	if not REFERENCE_DATA_PATH.exists():
		return pd.DataFrame()

	return pd.read_csv(REFERENCE_DATA_PATH)


def build_model_features(customer_input: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
	"""Convert raw form input into model-ready encoded feature dataframe."""
	base_row = {
		"tenure": float(customer_input["Tenure"]),
		"warehousetohome": float(customer_input["WarehouseToHome"]),
		"hourspendonapp": float(customer_input["HourSpendOnApp"]),
		"numberofdeviceregistered": float(customer_input["NumberOfDeviceRegistered"]),
		"orderamounthikefromlastyear": float(customer_input["OrderAmountHikeFromlastYear"]),
		"couponused": float(customer_input["CouponUsed"]),
		"ordercount": float(customer_input["OrderCount"]),
		"daysincelastorder": float(customer_input["DaySinceLastOrder"]),
		"cashbackamount": float(customer_input["CashbackAmount"]),
		"citytier": float(customer_input["CityTier"]),
		"gender": str(customer_input["Gender"]),
		"maritalstatus": str(customer_input["MaritalStatus"]),
		"preferredlogindevice": str(customer_input["PreferredLoginDevice"]),
		"preferredpaymentmode": str(customer_input["PreferredPaymentMode"]),
		"preferedordercat": str(customer_input["PreferedOrderCat"]),
	}

	working_df = pd.DataFrame([base_row])

	encoded_df = pd.get_dummies(
		working_df,
		columns=[
			"gender",
			"maritalstatus",
			"preferredlogindevice",
			"preferredpaymentmode",
			"preferedordercat",
		],
		drop_first=False,
		dtype=int,
	)

	model_df = encoded_df.reindex(columns=feature_columns, fill_value=0).astype(float)
	return model_df


def run_prediction(artifacts: dict[str, Any], model_features: pd.DataFrame) -> tuple[int, float | None]:
	"""Run churn prediction and probability using loaded model artifacts."""
	scaler = artifacts["scaler"]
	model = artifacts["model"]

	scaled_array = scaler.transform(model_features)
	scaled_df = pd.DataFrame(scaled_array, columns=model_features.columns)

	prediction = int(model.predict(scaled_df)[0])
	probability: float | None = None

	if hasattr(model, "predict_proba"):
		probability = float(model.predict_proba(scaled_df)[0][1])

	return prediction, probability


def build_retention_customer_series(customer_input: dict[str, Any], churn_prediction: int) -> pd.Series:
	"""Build series payload compatible with retention_agent functions."""
	return pd.Series(
		{
			"CustomerID": int(customer_input["CustomerID"]),
			"churn_prediction": float(churn_prediction),
			"Tenure": float(customer_input["Tenure"]),
			"PreferredLoginDevice": str(customer_input["PreferredLoginDevice"]),
			"CityTier": int(customer_input["CityTier"]),
			"WarehouseToHome": float(customer_input["WarehouseToHome"]),
			"PreferredPaymentMode": str(customer_input["PreferredPaymentMode"]),
			"Gender": str(customer_input["Gender"]),
			"HourSpendOnApp": float(customer_input["HourSpendOnApp"]),
			"NumberOfDeviceRegistered": float(customer_input["NumberOfDeviceRegistered"]),
			"PreferedOrderCat": str(customer_input["PreferedOrderCat"]),
			"SatisfactionScore": float(customer_input["SatisfactionScore"]),
			"MaritalStatus": str(customer_input["MaritalStatus"]),
			"NumberOfAddress": float(customer_input["NumberOfAddress"]),
			"Complain": float(customer_input["Complain"]),
			"OrderAmountHikeFromlastYear": float(customer_input["OrderAmountHikeFromlastYear"]),
			"CouponUsed": float(customer_input["CouponUsed"]),
			"OrderCount": float(customer_input["OrderCount"]),
			"DaySinceLastOrder": float(customer_input["DaySinceLastOrder"]),
			"CashbackAmount": float(customer_input["CashbackAmount"]),
		}
	)


def generate_retention_output(customer_series: pd.Series) -> tuple[list[str], str, str, str, str | None]:
	"""Generate retention explanation/message via genai retention agent logic."""
	risk_signals = identify_risk_signals(customer_series)
	if RETENTION_IMPORT_ERROR:
		explanation = build_fallback_explanation(risk_signals)
		message = build_fallback_retention_message(customer_series, risk_signals)
		return risk_signals, explanation, message, "fallback_local", RETENTION_IMPORT_ERROR

	try:
		api_token = get_hf_api_token()
		provider = resolve_provider(api_token, provider_override=None)
		client = create_hf_client(api_token, provider)

		explanation, message, status, error = generate_customer_outputs(
			client=client,
			model_id=DEFAULT_MODEL_ID,
			temperature=0.5,
			max_new_tokens=260,
			max_retries=1,
			initial_retry_delay=1.0,
			customer=customer_series,
			risk_signals=risk_signals,
		)

		return risk_signals, explanation, message, status, error

	except Exception as exc:
		explanation = build_fallback_explanation(risk_signals)
		message = build_fallback_retention_message(customer_series, risk_signals)
		return risk_signals, explanation, message, "fallback_local", str(exc)


def resolve_dashboard_url() -> str:
	"""Resolve a likely running dashboard URL for back-navigation."""
	# If dashboard URL is passed as a query param, honor it first.
	query_params = st.query_params
	if "dashboard_url" in query_params:
		return str(query_params["dashboard_url"])

	# Otherwise probe commonly used local dashboard ports.
	for port in (8501, 8511, 8513, 8515):
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
			sock.settimeout(0.2)
			if sock.connect_ex(("127.0.0.1", port)) == 0:
				return f"http://localhost:{port}"

	# Safe fallback.
	return "http://localhost:8501"


def main() -> None:
	"""Render Streamlit UI and handle churn prediction workflow."""
	st.set_page_config(page_title="Churn Predictor + Retention Message", layout="wide")
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');

		.stApp {
			background:
				radial-gradient(circle at 8% 8%, #e0f2fe 0%, transparent 33%),
				radial-gradient(circle at 95% 4%, #fef3c7 0%, transparent 28%),
				#f8fafc;
		}

		h1, h2, h3 {
			font-family: 'Space Grotesk', sans-serif;
		}

		p, li, div, span {
			font-family: 'Manrope', sans-serif;
		}

		.predict-hero {
			background: linear-gradient(135deg, #0f766e 0%, #155e75 55%, #1d4ed8 100%);
			color: #f8fafc;
			padding: 1.1rem 1.2rem;
			border-radius: 14px;
			border: 1px solid rgba(255,255,255,0.14);
			box-shadow: 0 10px 22px rgba(2, 132, 199, 0.22);
			margin-bottom: 0.85rem;
		}

		.predict-kicker {
			font-size: 0.74rem;
			text-transform: uppercase;
			letter-spacing: 1px;
			opacity: 0.88;
		}

		.predict-title {
			font-size: 1.65rem;
			font-weight: 700;
			margin: 0.25rem 0 0.2rem 0;
		}

		.predict-sub {
			margin: 0;
			opacity: 0.95;
		}

		.ai-panel {
			background: #ffffff;
			border: 1px solid #e2e8f0;
			border-radius: 12px;
			padding: 0.8rem 0.95rem;
			box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	dashboard_url = resolve_dashboard_url()
	st.link_button("← Back to Main Dashboard", dashboard_url)
	st.markdown(
		"""
		<div class="predict-hero">
			<div class="predict-kicker">Individual Customer Intelligence</div>
			<div class="predict-title">Customer Churn Prediction & AI Retention Assistant</div>
			<p class="predict-sub">Run one-customer churn scoring and receive an AI-ready retention recommendation with explainable risk signals.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	try:
		artifacts = load_model_artifacts()
	except Exception as exc:
		st.error(f"Model load error: {exc}")
		st.stop()

	reference_df = load_reference_data()

	st.subheader("Customer Input Form")
	st.caption("Fill all customer columns (except Churn target) and submit.")

	with st.form("customer_form"):
		col1, col2, col3 = st.columns(3)

		with col1:
			customer_id = st.number_input(
				"CustomerID",
				min_value=1,
				value=int(round(get_numeric_default(reference_df, "CustomerID"))),
				step=1,
			)
			tenure = st.number_input("Tenure", min_value=0.0, value=get_numeric_default(reference_df, "Tenure"), step=1.0)
			preferred_login_device = st.selectbox(
				"PreferredLoginDevice",
				options=get_category_options(reference_df, "PreferredLoginDevice"),
			)
			city_tier = st.number_input(
				"CityTier",
				min_value=1,
				max_value=5,
				value=int(round(get_numeric_default(reference_df, "CityTier"))),
				step=1,
			)
			warehouse_to_home = st.number_input(
				"WarehouseToHome",
				min_value=0.0,
				value=get_numeric_default(reference_df, "WarehouseToHome"),
				step=1.0,
			)
			preferred_payment_mode = st.selectbox(
				"PreferredPaymentMode",
				options=get_category_options(reference_df, "PreferredPaymentMode"),
			)
			gender = st.selectbox("Gender", options=get_category_options(reference_df, "Gender"))

		with col2:
			hour_spend_on_app = st.number_input(
				"HourSpendOnApp",
				min_value=0.0,
				value=get_numeric_default(reference_df, "HourSpendOnApp"),
				step=1.0,
			)
			number_of_device_registered = st.number_input(
				"NumberOfDeviceRegistered",
				min_value=0,
				value=int(round(get_numeric_default(reference_df, "NumberOfDeviceRegistered"))),
				step=1,
			)
			prefered_order_cat = st.selectbox(
				"PreferedOrderCat",
				options=get_category_options(reference_df, "PreferedOrderCat"),
			)
			satisfaction_score = st.number_input(
				"SatisfactionScore",
				min_value=0,
				max_value=5,
				value=int(round(get_numeric_default(reference_df, "SatisfactionScore"))),
				step=1,
			)
			marital_status = st.selectbox(
				"MaritalStatus",
				options=get_category_options(reference_df, "MaritalStatus"),
			)
			number_of_address = st.number_input(
				"NumberOfAddress",
				min_value=0,
				value=int(round(get_numeric_default(reference_df, "NumberOfAddress"))),
				step=1,
			)
			complain = st.selectbox("Complain", options=[0, 1], index=int(round(get_numeric_default(reference_df, "Complain"))))

		with col3:
			order_amount_hike = st.number_input(
				"OrderAmountHikeFromlastYear",
				min_value=0.0,
				value=get_numeric_default(reference_df, "OrderAmountHikeFromlastYear"),
				step=1.0,
			)
			coupon_used = st.number_input(
				"CouponUsed",
				min_value=0.0,
				value=get_numeric_default(reference_df, "CouponUsed"),
				step=1.0,
			)
			order_count = st.number_input(
				"OrderCount",
				min_value=0.0,
				value=get_numeric_default(reference_df, "OrderCount"),
				step=1.0,
			)
			day_since_last_order = st.number_input(
				"DaySinceLastOrder",
				min_value=0.0,
				value=get_numeric_default(reference_df, "DaySinceLastOrder"),
				step=1.0,
			)
			cashback_amount = st.number_input(
				"CashbackAmount",
				min_value=0.0,
				value=get_numeric_default(reference_df, "CashbackAmount"),
				step=1.0,
			)

		submit = st.form_submit_button("Predict Churn + Generate Retention Message", use_container_width=True)

	if not submit:
		st.info("Fill the form and click the button to run prediction.")
		return

	customer_input = {
		"CustomerID": customer_id,
		"Tenure": tenure,
		"PreferredLoginDevice": preferred_login_device,
		"CityTier": city_tier,
		"WarehouseToHome": warehouse_to_home,
		"PreferredPaymentMode": preferred_payment_mode,
		"Gender": gender,
		"HourSpendOnApp": hour_spend_on_app,
		"NumberOfDeviceRegistered": number_of_device_registered,
		"PreferedOrderCat": prefered_order_cat,
		"SatisfactionScore": satisfaction_score,
		"MaritalStatus": marital_status,
		"NumberOfAddress": number_of_address,
		"Complain": complain,
		"OrderAmountHikeFromlastYear": order_amount_hike,
		"CouponUsed": coupon_used,
		"OrderCount": order_count,
		"DaySinceLastOrder": day_since_last_order,
		"CashbackAmount": cashback_amount,
	}

	feature_columns = artifacts["feature_columns"]
	model_features = build_model_features(customer_input, feature_columns)
	prediction, probability = run_prediction(artifacts, model_features)

	st.subheader("Prediction Summary")
	label_text = "Will Churn" if prediction == 1 else "Will Not Churn"

	metric_col1, metric_col2 = st.columns(2)
	metric_col1.metric("Predicted Churn Label", label_text)
	if probability is not None:
		metric_col2.metric("Churn Probability", f"{probability:.2%}")
	else:
		metric_col2.metric("Churn Probability", "N/A")

	customer_series = build_retention_customer_series(customer_input, prediction)
	risk_signals, explanation, message, status, error = generate_retention_output(customer_series)

	risk_col, explain_col = st.columns(2)
	with risk_col:
		st.markdown("<div class='ai-panel'>", unsafe_allow_html=True)
		st.subheader("Detected Risk Signals")
		for signal in risk_signals:
			st.write(f"- {signal}")
		st.markdown("</div>", unsafe_allow_html=True)

	with explain_col:
		st.markdown("<div class='ai-panel'>", unsafe_allow_html=True)
		st.subheader("Churn Explanation")
		st.write(explanation)
		st.markdown("</div>", unsafe_allow_html=True)

	st.markdown("<div class='ai-panel'>", unsafe_allow_html=True)
	st.subheader("Personalized Retention Message")
	st.write(message)
	st.markdown("</div>", unsafe_allow_html=True)

	with st.expander("Generation Details"):
		st.write({
			"status": status,
			"error": error,
			"model": DEFAULT_MODEL_ID,
		})


if __name__ == "__main__":
	main()
