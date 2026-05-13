# src/utils.py

import os
import joblib
import pandas as pd


# ==========================================
# CREATE DIRECTORIES
# ==========================================
def create_directories():

    folders = [

        "outputs/metrics",
        "outputs/plots",
        "outputs/reports",
        "models"
    ]

    for folder in folders:

        os.makedirs(
            folder,
            exist_ok=True
        )


# ==========================================
# SAVE MODEL
# ==========================================
def save_model(model, filename):

    joblib.dump(model, filename)

    print(f"Model saved at: {filename}")


# ==========================================
# LOAD MODEL
# ==========================================
def load_model(filename):

    return joblib.load(filename)


# ==========================================
# SAVE METRICS
# ==========================================
def save_metrics(results_df, filename):

    results_df.to_csv(
        filename,
        index=False
    )

    print(f"Metrics saved at: {filename}")


# ==========================================
# SAVE DATAFRAME
# ==========================================
def save_dataframe(df, filename):

    df.to_csv(
        filename,
        index=False
    )

    print(f"Dataframe saved at: {filename}")


# ==========================================
# GENERATE REPORT
# ==========================================
def generate_report(
    results_df,
    best_model_name
):

    report_path = (
        "outputs/reports/final_report.txt"
    )

    with open(report_path, "w") as file:

        file.write(
            "DIABETES PREDICTION REPORT\n"
        )

        file.write("=" * 50)

        file.write("\n\n")

        file.write(
            results_df.to_string()
        )

        file.write("\n\n")

        file.write(
            f"Best Model: {best_model_name}"
        )

    print(
        f"Report generated at: {report_path}"
    )