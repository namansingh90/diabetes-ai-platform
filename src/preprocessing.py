# src/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ==========================================
# LOAD DATASET
# ==========================================
def load_dataset(path):

    return pd.read_csv(path)


# ==========================================
# HANDLE MISSING VALUES
# ==========================================
def handle_missing_values(df):

    X_cols = df.columns[:-1]

    for col in X_cols:

        df[col] = df[col].replace(
            0,
            df[col].median()
        )

    return df


# ==========================================
# CAP OUTLIERS USING IQR
# ==========================================
def cap_outliers(df):

    X_cols = df.columns[:-1]

    for col in X_cols:

        q1 = df[col].quantile(0.25)

        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr

        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower, upper)

    return df


# ==========================================
# FEATURE SCALING
# ==========================================
def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# ==========================================
# APPLY SMOTE
# ==========================================
def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_smote, y_train_smote