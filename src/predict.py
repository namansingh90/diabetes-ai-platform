# src/predict.py

import joblib
import numpy as np


# ==========================================
# LOAD MODEL & SCALER
# ==========================================
model = joblib.load(
    "models/diabetes_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_diabetes(input_data):

    input_array = np.array(input_data)

    input_array = input_array.reshape(1, -1)

    input_scaled = scaler.transform(input_array)

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

        "Prediction": result,

        "Probability": round(
            probability * 100,
            2
        )
    }


# ==========================================
# SAMPLE TEST
# ==========================================
if __name__ == "__main__":

    sample_data = [
        6,
        148,
        72,
        35,
        0,
        33.6,
        0.627,
        50
    ]

    result = predict_diabetes(sample_data)

    print(result)