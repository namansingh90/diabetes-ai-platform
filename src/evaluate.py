# src/evaluate.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# ==========================================
# EVALUATE MODEL
# ==========================================
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = None

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(
            y_test,
            y_prob
        )

    print("\n==============================")
    print(f"Model: {model.__class__.__name__}")
    print("==============================")

    print(classification_report(
        y_test,
        y_pred
    ))

    return {

        "Model": model.__class__.__name__,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1": round(f1, 4),

        "ROC-AUC": round(roc_auc, 4)
    }


# ==========================================
# COMPARE MODELS
# ==========================================
def compare_models(results_list):

    return pd.DataFrame(results_list)


# ==========================================
# PLOT METRIC
# ==========================================
def plot_metric(results_df, metric, save_path=None):

    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x="Model",
        y=metric,
        data=results_df
    )

    for container in ax.containers:

        ax.bar_label(container, fmt="%.2f")

    plt.title(f"{metric} Comparison")

    plt.xticks(rotation=15)

    plt.grid(
        axis='y',
        linestyle='--',
        alpha=0.5
    )

    if save_path:

        plt.savefig(save_path)

    plt.show()


# ==========================================
# CONFUSION MATRIX
# ==========================================
def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    save_path=None
):

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test
    )

    plt.title(
        f"{model.__class__.__name__} Confusion Matrix"
    )

    if save_path:

        plt.savefig(save_path)

    plt.show()


# ==========================================
# ROC CURVE
# ==========================================
def plot_roc_curve(
    model,
    X_test,
    y_test,
    save_path=None
):

    if hasattr(model, "predict_proba"):

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test
        )

        plt.title(
            f"{model.__class__.__name__} ROC Curve"
        )

        if save_path:

            plt.savefig(save_path)

        plt.show()