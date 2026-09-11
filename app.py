"""
Streamlit app for the Breast Cancer Prediction (Keras) project.

Run locally with:
    streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

# --------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------------------------------
# Load model, scaler, and feature order once (cached across reruns)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("breast_cancer_model.keras")
    scaler = joblib.load("scaler.joblib")
    with open("feature_names.json") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


@st.cache_data
def load_reference_stats():
    """Used only to set sensible slider min/max/default values."""
    df = pd.read_csv("breast_cancer_data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"])
    X = df.drop(columns=["diagnosis"])
    return X.describe().T  # index=feature, columns=count/mean/std/min/25%/50%/75%/max


model, scaler, feature_names = load_artifacts()
stats = load_reference_stats()

GROUPS = {
    "Mean values": [f for f in feature_names if f.endswith("_mean")],
    "Standard error values": [f for f in feature_names if f.endswith("_se")],
    "Worst (largest) values": [f for f in feature_names if f.endswith("_worst")],
}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🩺 Breast Cancer Prediction")
st.markdown(
    "A neural network (Keras / TensorFlow) trained on the "
    "**Breast Cancer Wisconsin (Diagnostic)** dataset predicts whether a tumor "
    "is **benign** or **malignant** from 30 cell-nuclei measurements taken from "
    "a digitized image of a fine needle aspirate (FNA)."
)
st.info(
    "⚠️ **Disclaimer:** This is a student / portfolio machine-learning project. "
    "It is **not** a medical device and must not be used for real diagnosis. "
    "Always consult a qualified healthcare professional.",
    icon="⚠️",
)

tab_manual, tab_batch, tab_about = st.tabs(
    ["🔢 Manual Input", "📄 Batch Predict (CSV)", "ℹ️ About the model"]
)

# --------------------------------------------------------------------------
# Helper: run a prediction given a single-row DataFrame of raw features
# --------------------------------------------------------------------------
def predict(raw_df: pd.DataFrame):
    raw_df = raw_df[feature_names]  # enforce correct column order
    scaled = scaler.transform(raw_df)
    prob_malignant = model.predict(scaled, verbose=0).ravel()
    pred_class = (prob_malignant >= 0.5).astype(int)
    return prob_malignant, pred_class

# --------------------------------------------------------------------------
# TAB 1 — Manual input via sliders
# --------------------------------------------------------------------------
with tab_manual:
    st.caption(
        "Sliders are pre-filled with the dataset's average values — adjust them, "
        "or click a quick-load button below, then hit **Predict**."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Load a random *benign*-like example"):
            st.session_state["_load"] = "benign"
    with col2:
        if st.button("Load a random *malignant*-like example"):
            st.session_state["_load"] = "malignant"

    # Precompute quick-load example rows from the raw dataset (cached)
    @st.cache_data
    def example_rows():
        df = pd.read_csv("breast_cancer_data.csv")
        df = df.drop(columns=["id", "Unnamed: 32"])
        benign_row = df[df["diagnosis"] == "B"].sample(1, random_state=1).iloc[0]
        malignant_row = df[df["diagnosis"] == "M"].sample(1, random_state=1).iloc[0]
        return benign_row, malignant_row

    benign_example, malignant_example = example_rows()
    loaded = st.session_state.get("_load")
    preset = None
    if loaded == "benign":
        preset = benign_example
    elif loaded == "malignant":
        preset = malignant_example

    values = {}
    for group_name, cols in GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Mean values")):
            grid = st.columns(2)
            for i, col in enumerate(cols):
                row = stats.loc[col]
                default = float(preset[col]) if preset is not None else float(row["mean"])
                lo = float(row["min"])
                hi = float(row["max"])
                step = (hi - lo) / 200 if hi > lo else 0.01
                with grid[i % 2]:
                    values[col] = st.slider(
                        col.replace("_", " "),
                        min_value=lo,
                        max_value=hi,
                        value=min(max(default, lo), hi),
                        step=step,
                        key=f"slider_{col}",
                    )

    st.divider()
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values])
        prob, pred = predict(input_df)
        prob, pred = float(prob[0]), int(pred[0])

        label = "Malignant" if pred == 1 else "Benign"
        confidence = prob if pred == 1 else 1 - prob

        c1, c2 = st.columns([1, 2])
        with c1:
            if pred == 1:
                st.error(f"### Prediction: {label}")
            else:
                st.success(f"### Prediction: {label}")
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        with c2:
            st.write("**Probability of malignancy**")
            st.progress(prob)
            st.caption(f"Raw sigmoid output: {prob:.4f} (threshold = 0.50)")

# --------------------------------------------------------------------------
# TAB 2 — Batch prediction from an uploaded CSV
# --------------------------------------------------------------------------
with tab_batch:
    st.caption(
        "Upload a CSV with the same 30 feature columns as the training data "
        "(column names must match exactly; `id` / `diagnosis` columns, if present, are ignored)."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    with st.expander("Need a sample file to try?"):
        sample = pd.read_csv("breast_cancer_data.csv").drop(
            columns=["Unnamed: 32", "diagnosis"]
        ).head(5)
        st.download_button(
            "Download a 5-row sample CSV",
            data=sample.to_csv(index=False).encode("utf-8"),
            file_name="sample_input.csv",
            mime="text/csv",
        )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            drop_cols = [c for c in ["id", "diagnosis", "Unnamed: 32"] if c in batch_df.columns]
            batch_df_clean = batch_df.drop(columns=drop_cols)
            missing = set(feature_names) - set(batch_df_clean.columns)
            if missing:
                st.error(f"Missing required columns: {sorted(missing)}")
            else:
                probs, preds = predict(batch_df_clean)
                result = batch_df.copy()
                result["prob_malignant"] = probs.round(4)
                result["prediction"] = np.where(preds == 1, "Malignant", "Benign")
                st.success(f"Predicted {len(result)} rows.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# --------------------------------------------------------------------------
# TAB 3 — About
# --------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### Pipeline
`load → explore → split (60/20/20) → scale (StandardScaler) → build (Keras MLP) → train (early stopping) → evaluate`

### Architecture
```
Input(30 features)
  → Dense(16, relu)
  → Dense(8, relu)
  → Dense(1, sigmoid)
```

### Training details
- Optimizer: Adam (lr = 0.001)
- Loss: binary cross-entropy
- Early stopping on validation loss (patience = 10, best weights restored)

### Result on held-out test set
96.5% test accuracy — see the training notebook for the full evaluation.
confusion matrix, precision/recall, and ROC curve.

### Files
- `notebook/brest_cancer_dl_project.ipynb` – original training notebook
- `breast_cancer_model.keras` – the trained network
- `scaler.joblib` – the fitted `StandardScaler` (must be reused at inference time)
- `feature_names.json` – exact column order the model expects
        """
    )
