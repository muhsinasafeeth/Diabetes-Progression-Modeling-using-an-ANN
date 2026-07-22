import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="centered"
)

# ── Load model & scaler ──────────────────────────────────
FEATURE_NAMES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

@st.cache_resource
def load_model():
    model = keras.models.load_model("diabetes_ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Header ───────────────────────────────────────────────
st.title("🩺 Diabetes Progression Predictor")
st.markdown("""
Enter the patient's clinical values below.
The ANN model will predict their **disease progression score** (25 – 346).
""")
st.divider()

# ── Input sliders ────────────────────────────────────────
st.subheader("📋 Patient Clinical Data")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (normalized)", -0.11, 0.11, 0.00, 0.001)
    sex = st.selectbox("Sex", options=[-0.044, 0.051],
                        format_func=lambda x: "Female" if x < 0 else "Male")
    bmi = st.slider("BMI (normalized)", -0.09, 0.18, 0.02, 0.001)
    bp = st.slider("Blood Pressure (normalized)", -0.11, 0.13, 0.00, 0.001)
    s1 = st.slider("Total Cholesterol s1", -0.13, 0.15, 0.00, 0.001)

with col2:
    s2 = st.slider("LDL Cholesterol s2", -0.12, 0.20, 0.00, 0.001)
    s3 = st.slider("HDL Cholesterol s3", -0.10, 0.18, 0.00, 0.001)
    s4 = st.slider("TCH Ratio s4", -0.08, 0.19, 0.00, 0.001)
    s5 = st.slider("Triglycerides s5", -0.13, 0.13, 0.00, 0.001)
    s6 = st.slider("Blood Sugar s6", -0.14, 0.14, 0.00, 0.001)

st.divider()

# ── Predict button ───────────────────────────────────────
if st.button("🔍 Predict Progression Score", use_container_width=True):
    # Build input as a DataFrame with feature names to match how the scaler was fit
    input_data = pd.DataFrame(
        [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]],
        columns=FEATURE_NAMES,
    )
    input_scaled = scaler.transform(input_data)

    # Run prediction
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    prediction = float(np.clip(prediction, 25, 346))

    # Display result
    st.subheader("📊 Prediction Result")
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Progression Score", f"{prediction:.1f}")
    col_r2.metric("Scale Range", "25 – 346")

    # Risk level
    if prediction < 100:
        risk = "🟢 Low"
    elif prediction < 200:
        risk = "🟡 Moderate"
    else:
        risk = "🔴 High"
    col_r3.metric("Risk Level", risk)

    # Progress bar
    st.progress(int((prediction - 25) / 321 * 100))
    st.caption(f"Score {prediction:.1f} out of 346 maximum")

    # Feature importance note
    st.info(
        "💡 **Key drivers of this prediction:** "
        f"BMI = {bmi:.3f} | Triglycerides (s5) = {s5:.3f} | "
        f"Blood Pressure = {bp:.3f}"
    )
