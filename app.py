import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .info-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #f97316;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/calorie_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_model()

# Header
st.title("🔥 Calories Burnt Prediction System")
st.markdown("Estimate personalized workout energy expenditure using Machine Learning based on physiological and biometric indicators.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Enter Workout & Biometric Parameters")

    c1, c2 = st.columns(2)
    with c1:
        gender_str = st.selectbox("Gender", ["Male", "Female"])
        gender = 0 if gender_str == "Male" else 1

        age = st.slider("Age (years)", min_value=12, max_value=85, value=25)
        height = st.slider("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)

    with c2:
        weight = st.slider("Weight (kg)", min_value=30.0, max_value=150.0, value=68.0, step=0.5)
        duration = st.slider("Workout Duration (minutes)", min_value=1.0, max_value=60.0, value=20.0, step=0.5)

    c3, c4 = st.columns(2)
    with c3:
        heart_rate = st.slider("Heart Rate (bpm)", min_value=60.0, max_value=180.0, value=110.0, step=1.0)
    with c4:
        body_temp = st.slider("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=38.5, step=0.1)

    predict_btn = st.button("⚡ Predict Calories Burnt", type="primary", use_container_width=True)

# Physiological Calculations
height_m = height / 100.0
bmi = weight / (height_m ** 2)

if bmi < 18.5:
    bmi_status = "Underweight"
    bmi_color = "#38bdf8"
elif bmi < 25.0:
    bmi_status = "Normal Weight"
    bmi_color = "#4ade80"
elif bmi < 30.0:
    bmi_status = "Overweight"
    bmi_color = "#fbbf24"
else:
    bmi_status = "Obese"
    bmi_color = "#f87171"

with col2:
    st.subheader("📊 Prediction & Health Analytics")

    input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])

    if predict_btn or model is not None:
        if model is not None:
            predicted_cal = float(model.predict(input_data)[0])
        else:
            # Fallback physiological formula if model.pkl hasn't been compiled yet
            predicted_cal = float((duration * (heart_rate * 0.052 + weight * 0.045 + (body_temp - 37.0) * 4.5)) - (age * 0.08))

        predicted_cal = max(1.0, predicted_cal)
        burn_rate = predicted_cal / duration

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Energy Expenditure</div>
            <div class="metric-value">{predicted_cal:.1f} <span style="font-size: 1.5rem;">kcal</span></div>
            <div class="metric-label">Average Burn Rate: <b>{burn_rate:.2f} kcal/min</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("#### 🏃 Workout Insights & Benchmarks")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Body Mass Index (BMI)", f"{bmi:.1f}", delta=bmi_status)
        with m2:
            st.metric("Heart Rate Reserve", f"{heart_rate:.0f} bpm", delta="Active Aerobic")

        st.write("##### Equivalent Food Intake Burned:")
        p_slices = predicted_cal / 285.0
        apples = predicted_cal / 95.0
        c1_b, c2_b = st.columns(2)
        c1_b.info(f"🍕 **{p_slices:.1f}** slice(s) of pizza")
        c2_b.info(f"🍏 **{apples:.1f}** medium apple(s)")

    else:
        st.info("Click **Predict Calories Burnt** to view energy expenditure estimates.")

st.markdown("---")
st.caption("🔬 Developed as a Machine Learning fitness estimation project utilizing biological and physiological regression modeling.")
