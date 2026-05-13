# src/train.py

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


# ==========================================
# GET MODELS
# ==========================================
def get_models():

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "SVM":
            SVC(
                probability=True
            ),

        "KNN":
            KNeighborsClassifier(),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "Neural Network":
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1000,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                eval_metric='logloss',
                random_state=42
            )
    }

    return models


# ==========================================
# TRAIN MODEL
# ==========================================
def train_model(model, X_train, y_train):

    model.fit(X_train, y_train)

    return model