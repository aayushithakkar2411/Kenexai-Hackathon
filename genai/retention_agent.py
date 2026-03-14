"""Generate churn explanations and retention offers from warehouse data.

What this script does:
1) Connects to Gold warehouse SQLite database.
2) Queries all churn-risk customers (where churn prediction = 1).
3) Combines customer behavioral data with churn prediction results.
4) Detects likely churn-risk signals from behavior features.
5) Uses LangChain + Hugging Face Inference API with Meta-Llama
   to generate:
   - short churn explanation
   - formal personalized retention message with incentive
6) Saves output into a file inside genai folder (CSV/TXT).

Environment:
- Uses token from genai/.env (HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)
- Optional provider in genai/.env:
  HF_PROVIDER=novita
"""

from __future__ import annotations

import ast
import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENAI_ENV_PATH = Path(__file__).resolve().parent / ".env"
DEFAULT_WAREHOUSE_DB_PATH = PROJECT_ROOT / "data" / "gold" / "warehouse.db"
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_OUTPUT_CSV_PATH = Path(__file__).resolve().parent / "retention_messages_generated.csv"
DEFAULT_OUTPUT_TXT_PATH = Path(__file__).resolve().parent / "retention_messages_generated.txt"
DEFAULT_MAX_CUSTOMERS = 2

SYSTEM_PROMPT = (
	"You are a formal customer-retention analyst for an e-commerce company. "
	"Use only the provided customer data and risk signals. Do not invent facts. "
	"Write in a natural, warm-professional tone that sounds human, not robotic."
)

USER_PROMPT_TEMPLATE = """
Customer data:
{customer_profile}

Detected churn-risk signals:
{risk_signals}

Generate output as STRICT JSON with exactly these keys:
{{
  "churn_explanation": "...",
  "retention_message": "..."
}}

Rules:
1) churn_explanation: 1-2 sentences, <= 45 words, formal, data-grounded.
2) retention_message: 50-60 words, friendly-formal, personalized, and natural.
3) retention_message must include one concrete promotion/incentive.
4) retention_message must include a clear call-to-action.
5) Avoid robotic/template phrases; write like a human relationship manager.
6) Return JSON only. No markdown or extra keys.
""".strip()


def configure_logging(level: int = logging.INFO) -> logging.Logger:
	"""Configure and return module logger."""
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)
	return logging.getLogger("genai.retention_agent")


logger = configure_logging()


def parse_args() -> argparse.Namespace:
	"""Parse command-line options."""
	parser = argparse.ArgumentParser(
		description="Generate warehouse-driven retention insights and messages."
	)
	parser.add_argument(
		"--warehouse-db",
		type=Path,
		default=DEFAULT_WAREHOUSE_DB_PATH,
		help="Path to warehouse SQLite database.",
	)
	parser.add_argument(
		"--model-id",
		type=str,
		default=DEFAULT_MODEL_ID,
		help="Hugging Face model id. Default: meta-llama/Meta-Llama-3-8B-Instruct.",
	)
	parser.add_argument(
		"--provider",
		type=str,
		default=None,
		help="Optional HF inference provider, for example novita.",
	)
	parser.add_argument(
		"--max-customers",
		type=int,
		default=DEFAULT_MAX_CUSTOMERS,
		help="Number of churn-risk customers to process (default: 2 for testing).",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.5,
		help="LLM temperature for generation.",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=280,
		help="Maximum generated tokens per customer.",
	)
	parser.add_argument(
		"--max-retries",
		type=int,
		default=2,
		help="Retry attempts after the first generation failure.",
	)
	parser.add_argument(
		"--initial-retry-delay",
		type=float,
		default=1.0,
		help="Initial retry delay in seconds (exponential backoff).",
	)
	parser.add_argument(
		"--inter-request-delay",
		type=float,
		default=0.2,
		help="Delay between customers in seconds.",
	)
	parser.add_argument(
		"--output-format",
		type=str,
		choices=("csv", "txt"),
		default="csv",
		help="Output file format in genai folder (csv or txt).",
	)
	parser.add_argument(
		"--output-file",
		type=Path,
		default=None,
		help="Optional output file path. Default is in genai folder based on format.",
	)
	return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
	"""Resolve output file path from CLI settings."""
	if args.output_file is not None:
		return args.output_file

	if args.output_format == "txt":
		return DEFAULT_OUTPUT_TXT_PATH

	return DEFAULT_OUTPUT_CSV_PATH


def get_hf_api_token() -> str:
	"""Load Hugging Face token from genai/.env or environment."""
	load_dotenv(dotenv_path=GENAI_ENV_PATH, override=True)

	api_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
	if not api_token:
		raise EnvironmentError(
			"Missing token. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in genai/.env."
		)

	os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
	return api_token


def resolve_provider(api_token: str, provider_override: str | None) -> str | None:
	"""Resolve provider from CLI/env/token style."""
	if provider_override:
		return provider_override

	env_provider = os.getenv("HF_PROVIDER")
	if env_provider:
		return env_provider

	if not api_token.startswith("hf_"):
		return "novita"

	return None


def create_hf_client(api_token: str, provider: str | None) -> InferenceClient:
	"""Create Hugging Face inference client."""
	if provider:
		logger.info("Using provider: %s", provider)
		return InferenceClient(provider=provider, api_key=api_token)

	logger.info("Using Hugging Face auto-routing provider selection.")
	return InferenceClient(api_key=api_token)


def map_langchain_role_to_hf_role(message_type: str) -> str:
	"""Map LangChain message types to HF chat roles."""
	if message_type == "system":
		return "system"
	if message_type == "ai":
		return "assistant"
	return "user"


class RetentionGenerationChain:
	"""Reusable LangChain wrapper around HF chat completions for retention generation."""

	def __init__(
		self,
		client: InferenceClient,
		model_id: str,
		temperature: float,
		max_new_tokens: int,
	) -> None:
		self.client = client
		self.model_id = model_id
		self.temperature = temperature
		self.max_new_tokens = max_new_tokens
		self.chain = self._build_chain()

	def _build_chain(self):
		prompt = ChatPromptTemplate.from_messages(
			[
				("system", SYSTEM_PROMPT),
				("human", USER_PROMPT_TEMPLATE),
			]
		)

		llm_runnable = RunnableLambda(self._invoke_hf_chat)
		return prompt | llm_runnable | StrOutputParser()

	def _invoke_hf_chat(self, prompt_value: Any) -> str:
		if hasattr(prompt_value, "messages"):
			messages = list(getattr(prompt_value, "messages"))
		elif isinstance(prompt_value, list):
			messages = prompt_value
		else:
			messages = [prompt_value]

		hf_messages: list[dict[str, str]] = []
		for message in messages:
			if isinstance(message, tuple) and len(message) == 2 and message[0] == "messages":
				nested_messages = message[1]
				if isinstance(nested_messages, list):
					messages.extend(nested_messages)
				continue

			message_type = str(getattr(message, "type", "human"))
			raw_content = getattr(message, "content", "")
			if isinstance(raw_content, list):
				parts: list[str] = []
				for part in raw_content:
					if isinstance(part, dict):
						text_part = part.get("text")
						if text_part:
							parts.append(str(text_part))
					else:
						parts.append(str(part))
				message_content = "\n".join(parts).strip()
			else:
				message_content = str(raw_content).strip()

			if not message_content:
				continue

			hf_messages.append(
				{
					"role": map_langchain_role_to_hf_role(message_type),
					"content": message_content,
				}
			)

		if not hf_messages:
			raise ValueError("No valid chat messages were produced from the prompt.")

		response = self.client.chat.completions.create(
			model=self.model_id,
			messages=hf_messages,
			temperature=self.temperature,
			max_tokens=self.max_new_tokens,
		)

		if response.choices and len(response.choices) > 0:
			return str(response.choices[0].message.content or "")

		return str(response)

	def invoke(self, customer_profile: str, risk_signals: str) -> str:
		return str(
			self.chain.invoke(
				{
					"customer_profile": customer_profile,
					"risk_signals": risk_signals,
				}
			)
		)


def connect_warehouse(db_path: Path) -> sqlite3.Connection:
	"""Open SQLite connection for the warehouse database."""
	if not db_path.exists():
		raise FileNotFoundError(f"Warehouse DB not found: {db_path}")

	conn = sqlite3.connect(db_path)
	conn.row_factory = sqlite3.Row
	return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
	"""Check if a table exists in SQLite DB."""
	query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
	row = conn.execute(query, (table_name,)).fetchone()
	return row is not None


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
	"""Return set of columns for a given table."""
	rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
	return {row[1] for row in rows}


def detect_prediction_source(conn: sqlite3.Connection) -> tuple[str | None, str | None]:
	"""Detect optional churn prediction table and prediction column.

	If none is found, the pipeline falls back to fact_customer_behavior.Churn.
	"""
	candidate_tables = (
		"churn_prediction_results",
		"churn_predictions",
		"ml_churn_predictions",
	)
	candidate_columns = (
		"churn_prediction",
		"prediction",
		"predicted_churn",
		"ChurnPrediction",
	)

	for table_name in candidate_tables:
		if not table_exists(conn, table_name):
			continue

		columns = get_table_columns(conn, table_name)
		if "CustomerID" not in columns:
			continue

		for prediction_column in candidate_columns:
			if prediction_column in columns:
				return table_name, prediction_column

	return None, None


def fetch_churn_risk_customers(
	conn: sqlite3.Connection,
	max_customers: int | None = None,
) -> pd.DataFrame:
	"""Fetch churn-risk customers by combining behavior + prediction data."""
	prediction_table, prediction_column = detect_prediction_source(conn)

	if prediction_table and prediction_column:
		logger.info(
			"Combining fact behavior with prediction table %s.%s",
			prediction_table,
			prediction_column,
		)

		prediction_expr = f"COALESCE(pred.{prediction_column}, f.Churn)"
		prediction_join = (
			f"LEFT JOIN {prediction_table} pred ON pred.CustomerID = c.CustomerID"
		)
	else:
		logger.info("No separate prediction table found. Using fact_customer_behavior.Churn")
		prediction_expr = "f.Churn"
		prediction_join = ""

	query = f"""
		SELECT
			c.customer_key,
			c.CustomerID,
			CAST({prediction_expr} AS REAL) AS churn_prediction,
			f.Tenure,
			f.WarehouseToHome,
			f.HourSpendOnApp,
			f.NumberOfDeviceRegistered,
			f.OrderAmountHikeFromlastYear,
			f.CouponUsed,
			f.OrderCount,
			f.DaySinceLastOrder,
			f.CashbackAmount,
			c.Gender,
			c.MaritalStatus,
			c.CityTier,
			d.PreferredLoginDevice,
			p.PreferredPaymentMode,
			o.PreferedOrderCat
		FROM fact_customer_behavior f
		INNER JOIN dim_customer c ON f.customer_key = c.customer_key
		LEFT JOIN dim_device d ON f.device_key = d.device_key
		LEFT JOIN dim_payment p ON f.payment_key = p.payment_key
		LEFT JOIN dim_order_category o ON f.category_key = o.category_key
		{prediction_join}
		WHERE CAST({prediction_expr} AS REAL) = 1
		ORDER BY c.CustomerID
	"""

	params: list[Any] = []
	if max_customers is not None:
		query += "\nLIMIT ?"
		params.append(max_customers)

	df = pd.read_sql_query(query, conn, params=params)

	if df.empty:
		raise ValueError("No churn-risk customers found (churn prediction = 1).")

	logger.info("Fetched churn-risk customers: %d", len(df))
	return df


def to_float(value: Any) -> float | None:
	"""Safe float conversion helper."""
	try:
		if value is None:
			return None
		if pd.isna(value):
			return None
		return float(value)
	except Exception:
		return None


def identify_risk_signals(customer: pd.Series) -> list[str]:
	"""Identify likely churn drivers from behavior features."""
	signals: list[str] = []

	day_since_last_order = to_float(customer.get("DaySinceLastOrder"))
	order_count = to_float(customer.get("OrderCount"))
	hour_spend = to_float(customer.get("HourSpendOnApp"))
	coupon_used = to_float(customer.get("CouponUsed"))
	warehouse_distance = to_float(customer.get("WarehouseToHome"))
	tenure = to_float(customer.get("Tenure"))
	cashback_amount = to_float(customer.get("CashbackAmount"))

	if day_since_last_order is not None and day_since_last_order >= 7:
		signals.append("High inactivity since last order")

	if order_count is not None and order_count <= 1:
		signals.append("Very low recent order count")

	if hour_spend is not None and hour_spend <= 2:
		signals.append("Low app engagement")

	if coupon_used is not None and coupon_used == 0:
		signals.append("No recent coupon usage")

	if warehouse_distance is not None and warehouse_distance >= 20:
		signals.append("Long warehouse-to-home distance")

	if tenure is not None and tenure <= 3:
		signals.append("Low relationship tenure")

	if cashback_amount is not None and cashback_amount < 130:
		signals.append("Low effective cashback value")

	if not signals:
		signals.append("General elevated churn-risk pattern from model output")

	return signals


def normalize_value(value: Any) -> Any:
	"""Normalize NaN values for profile text."""
	if value is None:
		return "NA"
	if isinstance(value, float) and pd.isna(value):
		return "NA"
	return value


def build_customer_profile(customer: pd.Series) -> str:
	"""Build compact customer profile text for prompting."""
	profile_map = {
		"CustomerID": normalize_value(customer.get("CustomerID")),
		"PredictedChurnLabel": normalize_value(customer.get("churn_prediction")),
		"TenureMonths": normalize_value(customer.get("Tenure")),
		"DaySinceLastOrder": normalize_value(customer.get("DaySinceLastOrder")),
		"OrderCountLastMonth": normalize_value(customer.get("OrderCount")),
		"HourSpendOnApp": normalize_value(customer.get("HourSpendOnApp")),
		"CouponUsed": normalize_value(customer.get("CouponUsed")),
		"CashbackAmount": normalize_value(customer.get("CashbackAmount")),
		"WarehouseToHome": normalize_value(customer.get("WarehouseToHome")),
		"OrderAmountHikeFromLastYear": normalize_value(
			customer.get("OrderAmountHikeFromlastYear")
		),
		"PreferredLoginDevice": normalize_value(customer.get("PreferredLoginDevice")),
		"PreferredPaymentMode": normalize_value(customer.get("PreferredPaymentMode")),
		"PreferredOrderCategory": normalize_value(customer.get("PreferedOrderCat")),
		"Gender": normalize_value(customer.get("Gender")),
		"MaritalStatus": normalize_value(customer.get("MaritalStatus")),
		"CityTier": normalize_value(customer.get("CityTier")),
	}

	return "\n".join([f"- {k}: {v}" for k, v in profile_map.items()])


def clean_generated_text(raw_text: str) -> str:
	"""Normalize model output text for downstream parsing."""
	text = str(raw_text or "").strip()
	text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
	text = re.sub(r"\s*```\s*$", "", text).strip()
	text = (
		text.replace("“", '"')
		.replace("”", '"')
		.replace("’", "'")
		.replace("‘", "'")
	)
	return text


def normalize_generated_payload(payload: dict[str, Any]) -> dict[str, str]:
	"""Normalize parsed payload into exact required keys."""
	normalized: dict[str, Any] = {}
	for key, value in payload.items():
		normalized_key = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
		normalized[normalized_key] = value

	def pick_value(aliases: tuple[str, ...]) -> str:
		for alias in aliases:
			value = normalized.get(alias)
			if value is None:
				continue
			value_text = re.sub(r"\s+", " ", str(value)).strip().strip('"').strip("'")
			if value_text:
				return value_text
		return ""

	explanation = pick_value(
		(
			"churn_explanation",
			"churn_reason",
			"churnreason",
			"explanation",
			"reason",
		)
	)
	retention_message = pick_value(
		(
			"retention_message",
			"retentionmessage",
			"message",
			"retention_offer",
			"offer",
		)
	)

	if not explanation or not retention_message:
		raise ValueError("Model output JSON missing required keys or values.")

	return {
		"churn_explanation": explanation,
		"retention_message": retention_message,
	}


def try_parse_json_like(text: str) -> dict[str, Any] | None:
	"""Attempt parsing JSON-like object text into a dictionary."""
	candidate = text.strip()
	if not candidate:
		return None

	for attempt_text in (
		candidate,
		re.sub(r",\s*([}\]])", r"\1", candidate),
	):
		try:
			parsed = json.loads(attempt_text)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			pass

	try:
		parsed_literal = ast.literal_eval(candidate)
		if isinstance(parsed_literal, dict):
			return parsed_literal
	except Exception:
		pass

	return None


def extract_balanced_json_objects(text: str) -> list[str]:
	"""Extract balanced brace-delimited objects from text."""
	objects: list[str] = []
	depth = 0
	start: int | None = None
	in_string = False
	string_quote = ""
	escaped = False

	for index, char in enumerate(text):
		if escaped:
			escaped = False
			continue

		if char == "\\":
			escaped = True
			continue

		if in_string:
			if char == string_quote:
				in_string = False
			continue

		if char in ('"', "'"):
			in_string = True
			string_quote = char
			continue

		if char == "{":
			if depth == 0:
				start = index
			depth += 1
			continue

		if char == "}" and depth > 0:
			depth -= 1
			if depth == 0 and start is not None:
				objects.append(text[start : index + 1])
				start = None

	return objects


def extract_fields_from_labeled_text(text: str) -> dict[str, str] | None:
	"""Extract required fields from non-JSON labeled model output."""
	explanation_match = re.search(
		r"(?is)(?:churn[_\s-]*explanation|churn[_\s-]*reason|explanation)\s*[:\-]\s*(.+?)(?=(?:retention[_\s-]*message|retention[_\s-]*offer|message)\s*[:\-])",
		text,
	)
	message_match = re.search(
		r"(?is)(?:retention[_\s-]*message|retention[_\s-]*offer|message)\s*[:\-]\s*(.+)$",
		text,
	)

	if not explanation_match or not message_match:
		return None

	explanation = re.sub(r"\s+", " ", explanation_match.group(1)).strip().strip('"').strip("'")
	retention_message = re.sub(r"\s+", " ", message_match.group(1)).strip().strip('"').strip("'")

	if not explanation or not retention_message:
		return None

	return {
		"churn_explanation": explanation,
		"retention_message": retention_message,
	}


def parse_json_from_text(raw_text: str) -> dict[str, Any]:
	"""Extract and parse model output into required churn/message keys."""
	text = clean_generated_text(raw_text)
	if not text:
		raise ValueError("Model output was empty.")

	candidate_objects = [text, *extract_balanced_json_objects(text)]
	seen_candidates: set[str] = set()

	for candidate in candidate_objects:
		candidate_text = candidate.strip()
		if not candidate_text or candidate_text in seen_candidates:
			continue
		seen_candidates.add(candidate_text)

		parsed_candidate = try_parse_json_like(candidate_text)
		if not parsed_candidate:
			continue

		try:
			return normalize_generated_payload(parsed_candidate)
		except ValueError:
			continue

	labeled_fields = extract_fields_from_labeled_text(text)
	if labeled_fields:
		return labeled_fields

	raise ValueError("Model output did not contain valid JSON object.")


def build_fallback_explanation(risk_signals: list[str]) -> str:
	"""Fallback explanation when LLM output fails."""
	core_signals = ", ".join(risk_signals[:3])
	return (
		"The customer shows elevated churn risk based on behavioral indicators, "
		f"including: {core_signals}."
	)


def build_fallback_retention_message(customer: pd.Series, risk_signals: list[str]) -> str:
	"""Fallback formal retention message when LLM output fails."""
	customer_id = normalize_value(customer.get("CustomerID"))
	primary_signal = risk_signals[0] if risk_signals else "recent engagement decline"
	return (
		f"Dear Customer {customer_id}, thank you for being with us. We noticed "
		f"{primary_signal.lower()} and want to help you stay engaged. Enjoy a 15% "
		"discount and priority support on your next purchase with code STAY15, valid "
		"for 7 days. Please place your next order this week, and our team will "
		"personally assist you."
	)


def generate_customer_outputs(
	client: InferenceClient,
	model_id: str,
	temperature: float,
	max_new_tokens: int,
	max_retries: int,
	initial_retry_delay: float,
	customer: pd.Series,
	risk_signals: list[str],
	generation_chain: RetentionGenerationChain | None = None,
) -> tuple[str, str, str, str | None]:
	"""Generate churn explanation and retention message for one customer.

	Returns:
		(churn_explanation, retention_message, generation_status, generation_error)
	"""
	customer_profile = build_customer_profile(customer)
	risk_signals_text = ", ".join(risk_signals)

	active_chain = generation_chain or RetentionGenerationChain(
		client=client,
		model_id=model_id,
		temperature=temperature,
		max_new_tokens=max_new_tokens,
	)

	for attempt in range(max_retries + 1):
		try:
			content = active_chain.invoke(
				customer_profile=customer_profile,
				risk_signals=risk_signals_text,
			)
			parsed = parse_json_from_text(content)

			explanation = str(parsed.get("churn_explanation", "")).strip()
			retention_message = str(parsed.get("retention_message", "")).strip()

			if not explanation or not retention_message:
				raise ValueError("Model output JSON missing required keys or values.")

			return explanation, retention_message, "success", None

		except Exception as exc:
			is_last_attempt = attempt >= max_retries
			if is_last_attempt:
				fallback_explanation = build_fallback_explanation(risk_signals)
				fallback_message = build_fallback_retention_message(customer, risk_signals)
				return fallback_explanation, fallback_message, "fallback", str(exc)

			delay = initial_retry_delay * (2 ** attempt)
			logger.warning(
				"Generation retry %d/%d after error: %s",
				attempt + 1,
				max_retries,
				exc,
			)
			time.sleep(delay)

	# Unreachable in normal flow.
	fallback_explanation = build_fallback_explanation(risk_signals)
	fallback_message = build_fallback_retention_message(customer, risk_signals)
	return fallback_explanation, fallback_message, "fallback", "Unknown generation state"


def generate_retention_results(
	customers_df: pd.DataFrame,
	client: InferenceClient,
	args: argparse.Namespace,
) -> pd.DataFrame:
	"""Generate outputs for all churn-risk customers."""
	rows: list[dict[str, Any]] = []
	total = len(customers_df)

	shared_chain = RetentionGenerationChain(
		client=client,
		model_id=args.model_id,
		temperature=args.temperature,
		max_new_tokens=args.max_new_tokens,
	)

	for idx, (_, customer) in enumerate(customers_df.iterrows(), start=1):
		customer_id = customer.get("CustomerID")
		risk_signals = identify_risk_signals(customer)

		explanation, message, status, error = generate_customer_outputs(
			client=client,
			model_id=args.model_id,
			temperature=args.temperature,
			max_new_tokens=args.max_new_tokens,
			max_retries=args.max_retries,
			initial_retry_delay=args.initial_retry_delay,
			customer=customer,
			risk_signals=risk_signals,
			generation_chain=shared_chain,
		)

		rows.append(
			{
				"customer_key": int(customer["customer_key"]),
				"CustomerID": int(customer["CustomerID"]),
				"churn_prediction": float(customer["churn_prediction"]),
				"detected_risk_signals": ", ".join(risk_signals),
				"churn_explanation": explanation,
				"retention_message": message,
				"generation_status": status,
				"generation_error": error,
				"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
			}
		)

		logger.info("Processed customer %s (%d/%d) with status=%s", customer_id, idx, total, status)

		if args.inter_request_delay > 0 and idx < total:
			time.sleep(args.inter_request_delay)

	return pd.DataFrame(rows)


def save_results_file(results_df: pd.DataFrame, output_path: Path, output_format: str) -> None:
	"""Save generated results to CSV or TXT file in genai folder."""
	if results_df.empty:
		raise ValueError("No results to save.")

	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_format == "txt":
		lines: list[str] = []
		for idx, row in results_df.reset_index(drop=True).iterrows():
			lines.append(f"Customer #{idx + 1}")
			lines.append(f"CustomerID: {row['CustomerID']}")
			lines.append(f"ChurnPrediction: {row['churn_prediction']}")
			lines.append(f"DetectedRiskSignals: {row['detected_risk_signals']}")
			lines.append(f"ChurnExplanation: {row['churn_explanation']}")
			lines.append(f"RetentionMessage: {row['retention_message']}")
			lines.append(f"GenerationStatus: {row['generation_status']}")
			lines.append(f"GeneratedAtUTC: {row['generated_at_utc']}")
			lines.append("-" * 80)

		output_path.write_text("\n".join(lines), encoding="utf-8")
	else:
		export_df = results_df[
			[
				"CustomerID",
				"churn_prediction",
				"detected_risk_signals",
				"churn_explanation",
				"retention_message",
				"generation_status",
				"generation_error",
				"generated_at_utc",
			]
		].copy()
		export_df.to_csv(output_path, index=False)

	logger.info("Saved %d rows to output file: %s", len(results_df), output_path)


def run() -> None:
	"""Run full GenAI retention workflow."""
	args = parse_args()

	logger.info("Starting GenAI retention workflow.")
	logger.info("Warehouse DB: %s", args.warehouse_db)
	output_path = resolve_output_path(args)
	logger.info("Output file: %s", output_path)

	api_token = get_hf_api_token()
	provider = resolve_provider(api_token, args.provider)
	client = create_hf_client(api_token, provider)

	with connect_warehouse(args.warehouse_db) as conn:
		customers_df = fetch_churn_risk_customers(conn, max_customers=args.max_customers)
		results_df = generate_retention_results(customers_df, client, args)

	save_results_file(results_df, output_path, args.output_format)

	logger.info("Completed GenAI retention workflow successfully.")


if __name__ == "__main__":
	run()
