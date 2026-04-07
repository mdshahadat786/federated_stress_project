import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Stress Detection", layout="centered")

# Title and introduction
st.title("Student Stress Detection System")
st.write("Please share your daily routine and feelings. I will help you understand your current stress level.")

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    st.success("System is ready to help you! ✅")
except:
    st.error("Model file not found ❌")
    st.stop()

# Input Section
st.header("Enter Student Details")

# Columns layout
col1, col2 = st.columns(2)

with col1:
    study = st.slider("Study Hours", 0, 15, 5)
    sleep = st.slider("Sleep Hours", 0, 15, 7)
    social = st.slider("Social Media Usage", 0, 10, 2)
    activity = st.slider("Physical Activity", 0, 10, 1)
    extra = st.slider("Extracurricular Activities", 0, 10, 1)

with col2:
    pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    family = st.slider("Family Support (1-5)", 1, 5, 4)
    screen = st.slider("Total Screen Time", 0, 15, 4)
    financial = st.slider("Financial Stress (1-5)", 1, 5, 2)
    examfear = st.slider("Exam Anxiety (1-5)", 1, 5, 2)
    timemanagement = st.slider("Time Management (1-5)", 1, 5, 3)

# Calculate total hours
total_hours = study + sleep + social + activity + extra

# Store current level
current_level = None

# Prediction
if st.button("Predict Stress"):

    if total_hours > 24:
        st.warning(f"⚠️ **Data Alert:** You have logged {total_hours} hours. A day only has 24 hours.")
    else:
        data_input = np.array([[study, sleep, social, pressure, family,
                                activity, screen, extra, financial,
                                examfear, timemanagement]])

        # Model prediction (optional)
        prediction_val = model.predict(data_input)[0]

        # Stress score calculation
        raw_score = (pressure * 10) + (examfear * 10) + (financial * 10) - (sleep * 5) - (family * 5)
        stress_score = max(0, min(100, raw_score + 40)) 

        st.markdown("---")
        st.subheader("Analysis Results")

        # ✅ FINAL LOGIC
        if stress_score <= 30:
            status = "Not Stressed"
            current_level = "Low"
            st.success(f"Stress Level: Low ({stress_score}%)")
            st.write("You are doing well!")

        elif stress_score <= 65:
            status = "Stressed"
            current_level = "Moderate"
            st.warning(f"Stress Level: Moderate ({stress_score}%)")
            st.write("You are under some pressure. Take it easy.")

        else:
            status = "Stressed"
            current_level = "High"
            st.error(f"Stress Level: High ({stress_score}%)")
            st.write("Please take a break and seek support.")

        # System Observation
        st.info(f"📋 **System Observation:** Student is **{status}**")

        # Progress bar
        st.progress(int(stress_score))

        # Suggestions
        st.subheader("Personalized Suggestions")
        if sleep < 6: 
            st.write("• Maintain a proper sleep schedule (at least 7-8 hours).")
        if study > 8: 
            st.write("• Take regular breaks during study time.")
        if pressure > 3: 
            st.write("• Break your academic tasks into smaller steps.")
        if financial > 3: 
            st.write("• Discuss financial concerns with a trusted person.")
        if examfear > 3: 
            st.write("• Practice mock tests to build confidence.")
        if timemanagement < 3: 
            st.write("• Follow a structured daily schedule.")

# -----------------------------
# 📊 Stress Level Overview
# -----------------------------
st.markdown("---")
st.subheader("📊 Stress Level Overview")

if current_level:
    levels = {
        "Low": 0,
        "Moderate": 0,
        "High": 0
    }

    levels[current_level] = 1

    chart_data = pd.DataFrame(
        list(levels.items()),
        columns=["Stress Level", "Value"]
    )

    st.bar_chart(chart_data.set_index("Stress Level"))
else:
    st.info("Run prediction to see graph")