import streamlit as st
import requests


st.title("Diabetes Prediction System")

st.write("Enter patient details")


# =====================================
# INPUTS
# =====================================

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0
)

glucose = st.number_input(
    "Glucose"
)

blood_pressure = st.number_input(
    "Blood Pressure"
)

skin_thickness = st.number_input(
    "Skin Thickness"
)

insulin = st.number_input(
    "Insulin"
)

bmi = st.number_input(
    "BMI"
)

dpf = st.number_input(
    "Diabetes Pedigree Function"
)

age = st.number_input(
    "Age",
    min_value=1
)


# =====================================
# PREDICT BUTTON
# =====================================

if st.button("Predict"):

    data = {

        "Pregnancies": pregnancies,

        "Glucose": glucose,

        "BloodPressure": blood_pressure,

        "SkinThickness": skin_thickness,

        "Insulin": insulin,

        "BMI": bmi,

        "DiabetesPedigreeFunction": dpf,

        "Age": age
    }

    response = requests.post(
        "https://diabetes-ai-platform.onrender.com/predict",
        json=data
    )

    result = response.json()

    st.subheader("Prediction")

    st.write(
        f"Result: {result['prediction']}"
    )

    st.write(
        f"Probability: {result['probability']}%"
    )