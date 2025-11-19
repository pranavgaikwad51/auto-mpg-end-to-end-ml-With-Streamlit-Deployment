import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Auto MPG Predictor", page_icon="🚗", layout="centered")

# ------------------------------
# Load Model (LOCAL ONLY)
# ------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Auto-mpg_best_model.pkl")

model = load_model()

# -----------------------
# Sidebar (Simple & Fast)
# -----------------------
with st.sidebar:
    st.title("Auto MPG Predictor")
    st.write("Dataset: UCI Auto MPG")
    st.write("Model: Trained Regression Model")
    st.markdown("---")
    st.write("**Input Features:**")
    st.write("- Acceleration")
    st.write("- Horsepower")
    st.write("- Weight")
    st.markdown("---")
    st.caption("Built by Pranav Gaikwad")

# -----------------------
# Main UI
# -----------------------
st.header("🚗 Predict Fuel Efficiency (MPG)")

acc = st.slider("Acceleration", 6.0, 30.0, 15.0)
hp = st.slider("Horsepower", 40, 400, 100)
wt = st.slider("Weight (lbs)", 1500, 6000, 3000)

input_df = pd.DataFrame({
    "acceleration": [acc],
    "horsepower": [hp],
    "weight": [wt]
})

if st.button("Predict MPG"):
    pred = model.predict(input_df)[0]
    st.success(f"Predicted MPG: {pred:.2f}")
