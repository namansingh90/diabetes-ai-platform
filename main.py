# main.py

from sklearn.model_selection import train_test_split

from src.preprocessing import *
from src.train import *
from src.evaluate import *
from src.utils import *


# ==========================================
# CREATE DIRECTORIES
# ==========================================
create_directories()


# ==========================================
# LOAD DATA
# ==========================================
print("\nLoading dataset...")

df = load_dataset(
    "data/diabetes.csv"
)

print(df.head())


# ==========================================
# PREPROCESSING
# ==========================================
print("\nHandling missing values...")

df = handle_missing_values(df)

print("\nCapping outliers...")

df = cap_outliers(df)


# ==========================================
# FEATURES & TARGET
# ==========================================
X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# ==========================================
# SCALING
# ==========================================
X_train_scaled, X_test_scaled, scaler = scale_features(

    X_train,
    X_test
)


# ==========================================
# APPLY SMOTE
# ==========================================
X_train_smote, y_train_smote = apply_smote(

    X_train_scaled,
    y_train
)


# ==========================================
# TRAINING
# ==========================================
models = get_models()

results = []

best_model = None

best_f1 = 0

best_model_name = ""


print("\nTraining models...\n")


for name, model in models.items():

    print(f"\nTraining {name}...")

    trained_model = train_model(

        model,

        X_train_smote,

        y_train_smote
    )

    result = evaluate_model(

        trained_model,

        X_test_scaled,

        y_test
    )

    results.append(result)

    if result["F1"] > best_f1:

        best_f1 = result["F1"]

        best_model = trained_model

        best_model_name = name


# ==========================================
# RESULTS
# ==========================================
results_df = compare_models(results)

print("\n===================================")

print("MODEL COMPARISON")

print("===================================\n")

print(results_df)


# ==========================================
# SAVE METRICS
# ==========================================
save_metrics(

    results_df,

    "outputs/metrics/model_results.csv"
)


# ==========================================
# PLOTS
# ==========================================
plot_metric(

    results_df,

    "Accuracy",

    "outputs/plots/accuracy_plot.png"
)

plot_metric(

    results_df,

    "F1",

    "outputs/plots/f1_plot.png"
)


# ==========================================
# CONFUSION MATRIX
# ==========================================
plot_confusion_matrix(

    best_model,

    X_test_scaled,

    y_test,

    "outputs/plots/confusion_matrix.png"
)


# ==========================================
# ROC CURVE
# ==========================================
plot_roc_curve(

    best_model,

    X_test_scaled,

    y_test,

    "outputs/plots/roc_curve.png"
)


# ==========================================
# SAVE BEST MODEL
# ==========================================
save_model(

    best_model,

    "models/diabetes_model.pkl"
)

save_model(

    scaler,

    "models/scaler.pkl"
)


# ==========================================
# GENERATE REPORT
# ==========================================
generate_report(

    results_df,

    best_model_name
)


print("\n===================================")

print("BEST MODEL:", best_model_name)

print("===================================")

print("\nPIPELINE EXECUTED SUCCESSFULLY")