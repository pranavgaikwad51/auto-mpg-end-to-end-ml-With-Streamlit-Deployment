import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import os

# --------------------------
# CONFIG
# --------------------------
MODEL_GITHUB_RAW = (
    "https://raw.githubusercontent.com/pranavgaikwad51/auto-mpg-end-to-end-ml-With-Streamlit-Deployment/"
    "main/Auto-mpg_best_model.pkl"
)
LOCAL_MODEL_PATH = "Auto-mpg_best_model.pkl"

st.set_page_config(
    page_title="Auto MPG Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --------------------------
# Model Loader
# --------------------------

def load_model(local_path=LOCAL_MODEL_PATH, github_raw_url=MODEL_GITHUB_RAW):
    if os.path.exists(local_path):
        try:
            return joblib.load(local_path)
        except Exception as e:
            st.warning(f"Failed to load local model: {e}")

    try:
        r = requests.get(github_raw_url, timeout=10)
        r.raise_for_status()
        return joblib.load(BytesIO(r.content))
    except Exception as e:
        raise RuntimeError(f"Could not load model: {e}")


@st.cache_resource
def get_model():
    return load_model()


# --------------------------
# Sidebar
# --------------------------
with st.sidebar:
    st.title("Auto MPG Predictor")
    st.markdown("**Project:** End-to-end ML — EDA → Modeling → Deployment")
    st.markdown("**Dataset:** [Auto MPG Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset)")
    st.markdown("**Repository:** [GitHub](https://github.com/pranavgaikwad51/auto-mpg-end-to-end-ml-With-Streamlit-Deployment)")
    st.markdown("---")
    st.subheader("Model Inputs:")
    st.markdown("- Acceleration\n- Horsepower\n- Weight")
    st.markdown("---")
    st.subheader("How to use:")
    st.markdown("1. Adjust sliders.\n2. Click Predict.\n3. View MPG output.")
    st.markdown("---")
    st.caption("Built by Pranav — Streamlit Deployment")


# --------------------------
# Main Page
# --------------------------
st.header("🚗 Auto MPG Predictor")
st.write("Predict car fuel efficiency (MPG) using a trained ML model.")

# Sliders
col1, col2, col3 = st.columns(3)
with col1:
    acceleration = st.slider("Acceleration (0-60 mph time)", 6.0, 30.0, 15.0, 0.1)
with col2:
    horsepower = st.slider("Horsepower", 40, 400, 100, 1)
with col3:
    weight = st.slider("Weight (lbs)", 1500, 6000, 3000, 10)

# Load model
try:
    model = get_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Predict
input_df = pd.DataFrame({
    "acceleration": [acceleration],
    "horsepower": [horsepower],
    "weight": [weight],
})

if st.button("Predict MPG"):
    try:
        pred = model.predict(input_df)[0]
        st.metric("Predicted MPG", f"{pred:.2f}")
        st.success(f"Estimated MPG: {pred:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Batch prediction optional
st.markdown("---")
st.subheader("Batch prediction (Optional)")
uploaded_file = st.file_uploader("Upload CSV (acceleration, horsepower, weight)", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        needed_cols = ["acceleration", "horsepower", "weight"]

        if not all(col in batch_df.columns for col in needed_cols):
            st.error(f"CSV must include: {needed_cols}")
        else:
            batch_df["predicted_mpg"] = model.predict(batch_df[needed_cols])
            st.dataframe(batch_df.head())

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv, "predictions.csv")
    except Exception as e:
        st.error(f"Upload failed: {e}")

st.markdown("---")
st.write("Model file:")
st.write(MODEL_GITHUB_RAW)
