from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Churn Predictor", layout="wide")


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "churn_model.pkl"


@st.cache_resource
def load_artifact(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    with path.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, dict):
        raise TypeError("Expected pickle artifact to be a dict.")

    required_keys = {"model", "feature_columns"}
    missing = required_keys.difference(artifact.keys())
    if missing:
        raise KeyError(f"Missing artifact keys: {sorted(missing)}")

    return artifact


def get_category_options(feature_columns: list[str], prefix: str) -> list[str]:
    options = [col[len(prefix) :] for col in feature_columns if col.startswith(prefix)]
    return sorted(options)


def build_feature_frame(
    feature_columns: list[str],
    numeric_inputs: dict[str, float],
    categorical_inputs: dict[str, str],
) -> pd.DataFrame:
    row = pd.Series(0.0, index=feature_columns, dtype="float64")

    for col, value in numeric_inputs.items():
        if col in row.index:
            row[col] = float(value)

    for prefix, selected_value in categorical_inputs.items():
        if selected_value == "Baseline/Other":
            continue
        dummy_col = f"{prefix}{selected_value}"
        if dummy_col in row.index:
            row[dummy_col] = 1.0

    return row.to_frame().T


def main() -> None:
    st.title("Customer Churn Prediction App")
    st.caption("Uses the trained artifact from ml/churn_model.pkl (churn model + elbow-based KMeans)")

    with st.sidebar:
        st.header("Model Artifact")
        model_path = st.text_input("Path to .pkl", value=str(DEFAULT_MODEL_PATH))

    try:
        artifact = load_artifact(model_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model artifact: {exc}")
        st.stop()

    model = artifact["model"]
    scaler = artifact.get("scaler")
    kmeans = artifact.get("kmeans")
    feature_columns = artifact["feature_columns"]
    cluster_customer_type_map = artifact.get("cluster_customer_type_map", {})

    # Normalize mapping keys so display and lookup are consistent.
    normalized_cluster_map = {
        int(k): str(v)
        for k, v in cluster_customer_type_map.items()
        if str(k).lstrip("-").isdigit()
    }

    model_name = model.__class__.__name__
    st.success(f"Loaded model: {model_name}")

    if normalized_cluster_map:
        with st.expander("Cluster Personas", expanded=False):
            persona_rows = [
                {
                    "cluster_name": f"Cluster {cluster_id + 1}",
                    "customer_type": normalized_cluster_map[cluster_id],
                }
                for cluster_id in sorted(normalized_cluster_map)
            ]
            st.dataframe(pd.DataFrame(persona_rows), use_container_width=True, hide_index=True)

    numeric_defaults = {
        "tenure": 12.0,
        "warehousetohome": 12.0,
        "hourspendonapp": 2.0,
        "numberofdeviceregistered": 3.0,
        "orderamounthikefromlastyear": 10.0,
        "couponused": 1.0,
        "ordercount": 3.0,
        "daysincelastorder": 6.0,
        "cashbackamount": 150.0,
        "citytier": 2.0,
    }

    st.subheader("Enter Customer Profile")

    col1, col2, col3 = st.columns(3)
    numeric_inputs: dict[str, float] = {}

    numeric_fields = [col for col in numeric_defaults if col in feature_columns]
    for idx, feature in enumerate(numeric_fields):
        target_col = [col1, col2, col3][idx % 3]
        with target_col:
            numeric_inputs[feature] = st.number_input(
                feature,
                value=float(numeric_defaults[feature]),
                step=1.0,
                format="%.2f",
            )

    categorical_prefixes = {
        "gender_": "Gender",
        "maritalstatus_": "Marital Status",
        "preferredlogindevice_": "Preferred Login Device",
        "preferredpaymentmode_": "Preferred Payment Mode",
        "preferedordercat_": "Preferred Order Category",
    }

    st.subheader("Categorical Attributes")
    categorical_inputs: dict[str, str] = {}
    c1, c2 = st.columns(2)

    prefix_items = list(categorical_prefixes.items())
    for idx, (prefix, label) in enumerate(prefix_items):
        options = get_category_options(feature_columns, prefix)
        if not options:
            continue

        widget_col = c1 if idx % 2 == 0 else c2
        with widget_col:
            categorical_inputs[prefix] = st.selectbox(
                label,
                options=["Baseline/Other", *options],
                index=0,
            )

    threshold = st.slider("Risk Threshold", min_value=0.0, max_value=1.0, value=0.70, step=0.01)

    use_scaler_default = scaler is not None and "LogisticRegression" in model_name
    use_scaler = st.checkbox(
        "Apply scaler before prediction",
        value=use_scaler_default,
        help="Enable for scaled models like Logistic Regression. Keep disabled for tree models.",
    )

    if st.button("Predict Churn", use_container_width=True):
        x_row = build_feature_frame(feature_columns, numeric_inputs, categorical_inputs)

        x_eval = x_row
        if use_scaler and scaler is not None:
            x_eval = pd.DataFrame(
                scaler.transform(x_row),
                columns=x_row.columns,
                index=x_row.index,
            )

        if not hasattr(model, "predict_proba"):
            st.error("Loaded model does not expose predict_proba().")
            st.stop()

        churn_probability = float(model.predict_proba(x_eval)[0, 1])
        churn_probability = float(np.clip(churn_probability, 0.0, 1.0))

        risk_label = "High churn risk" if churn_probability >= threshold else "Low churn risk"

        predicted_cluster_name = "Not available"
        predicted_customer_type = "Not available"
        if kmeans is not None:
            # KMeans in this project was trained on scaled features.
            if scaler is None:
                st.error("KMeans is available, but scaler is missing in artifact. Re-export model from notebook.")
                st.stop()

            x_cluster_eval = pd.DataFrame(
                scaler.transform(x_row),
                columns=x_row.columns,
                index=x_row.index,
            )

            cluster_idx = int(kmeans.predict(x_cluster_eval)[0])
            predicted_cluster_name = f"Cluster {cluster_idx + 1}"
            predicted_customer_type = normalized_cluster_map.get(
                cluster_idx,
                cluster_customer_type_map.get(str(cluster_idx), "Unknown"),
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Churn Probability", f"{churn_probability:.4f}")
        m2.metric("Risk Label", risk_label)
        m3.metric("Predicted Cluster", predicted_cluster_name)
        m4.metric("Customer Type", predicted_customer_type)

        st.progress(churn_probability)

        st.markdown("### Model Input Vector")
        st.dataframe(x_row, use_container_width=True)

    with st.expander("Artifact Metadata", expanded=False):
        st.write({
            "feature_count": len(feature_columns),
            "has_scaler": scaler is not None,
            "has_kmeans": kmeans is not None,
            "cluster_customer_type_map": normalized_cluster_map or artifact.get("cluster_customer_type_map", {}),
        })


if __name__ == "__main__":
    main()
