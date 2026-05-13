from fastapi import FastAPI
import joblib
import numpy as np

from app.schemas import DiabetesInput


# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("models/diabetes_model.pkl")

scaler = joblib.load("models/scaler.pkl")


# =====================================
# CREATE APP
# =====================================

app = FastAPI(
    title="Diabetes Prediction API"
)


# =====================================
# HOME ROUTE
# =====================================

@app.get("/")
def home():

    return {
        "message": "Diabetes Prediction API Running"
    }


# =====================================
# PREDICTION ROUTE
# =====================================

@app.post("/predict")
def predict(data: DiabetesInput):

    input_data = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    result = (
        "Diabetic"
        if prediction == 1
        else "Non-Diabetic"
    )

    return {

        "prediction": result,

        "probability": round(
            probability * 100,
            2
        )
    }