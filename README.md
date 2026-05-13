# Diabetes Prediction System

A Machine Learning based web application that predicts whether a person is diabetic or non-diabetic using medical attributes such as glucose level, BMI, insulin, age, etc.

The project includes:

- Data preprocessing
- Model training & evaluation
- SMOTE balancing
- Multiple ML model comparison
- FastAPI backend
- Streamlit frontend
- Cloud deployment

---

# Live Demo

## Frontend (Streamlit)

https://diabetes-ai-platform-mqqdxbddcyxfvhqibk8n3x.streamlit.app/

## Backend API (Render)

https://diabetes-ai-platform.onrender.com

---

# Project Structure

```bash
diabetes-ai-platform/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app/
│   └── api.py
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md