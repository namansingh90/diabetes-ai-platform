# Diabetes AI Platform

A complete end-to-end Machine Learning application for predicting diabetes using patient medical data.

This project includes:

- Data preprocessing
- Feature engineering
- Model training & evaluation
- Hyperparameter tuning
- Frontend using Streamlit
- Backend API using FastAPI
- Deployment-ready architecture

---

# Features

## Machine Learning

- Missing value handling
- Outlier capping using IQR
- Feature scaling
- SMOTE for class imbalance
- Multiple ML model comparison
- Hyperparameter tuning
- ROC Curve & Confusion Matrix
- Feature importance analysis
- PCA & LDA visualization

---

## Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Neural Network (MLPClassifier)
- XGBoost

---

## Frontend

Built using Streamlit.

Features:

- User-friendly form input
- Real-time prediction
- Prediction probability display

---

## Backend API

Built using FastAPI.

Features:

- REST API endpoints
- Swagger UI documentation
- JSON-based prediction API

---

# Project Structure

```bash
diabetes-ai-platform/
│
├── app/
│   ├── api.py
│   └── schemas.py
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── model_comparison.ipynb
│
├── outputs/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
