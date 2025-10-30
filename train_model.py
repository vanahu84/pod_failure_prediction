"""
train_model.py — Train pipeline for Next Pod Failure Prediction
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "pod_failure_prediction_dataset.csv"
MODEL_PATH = "pod_failure_predictor.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Helpers
# -----------------------------
def detect_target_column(df):
    # Accept a few common target names used earlier
    for name in ["predicted_pod_failure", "next_pod_failure", "target", "failure"]:
        if name in df.columns:
            return name
    raise ValueError("No target column found. Expected one of: predicted_pod_failure, next_pod_failure, target, failure")

# -----------------------------
# Load
# -----------------------------
print("Loading dataset from:", DATA_PATH)
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# -----------------------------
# Target detection
# -----------------------------
TARGET = detect_target_column(df)
print("Detected target column:", TARGET)

# Drop non-feature columns if present
X = df.drop(columns=[TARGET, "timestamp"] if "timestamp" in df.columns else [TARGET])
y = df[TARGET]

# Identify categorical and numerical features
# Treat low-cardinality object/string columns as categorical
categorical_features = [c for c in X.columns if X[c].dtype == "object" or X[c].dtype.name == "category"]
# Also include known categorical column names
for known_cat in ["autoscaler_action", "auto_scaler_triggered"]:
    if known_cat in X.columns and known_cat not in categorical_features:
        categorical_features.append(known_cat)

numerical_features = [c for c in X.columns if c not in categorical_features]
print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.nunique() > 1 else None
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# -----------------------------
# Candidate models
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=4),
}

# -----------------------------
# Quick hyperparameter search space for RandomizedSearch
# -----------------------------
param_spaces = {
    "RandomForest": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [None, 10, 20, 40],
        "classifier__min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 400],
        "classifier__max_depth": [3, 6, 10],
        "classifier__learning_rate": [0.01, 0.05, 0.1]
    }
}

# -----------------------------
# Train and evaluate
# -----------------------------
results = {}
for name, base_model in models.items():
    print(f"\nTraining: {name}")
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", base_model)])

    if name in param_spaces:
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_spaces[name],
            n_iter=6,
            cv=3,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        clf = search.best_estimator_
        print("Best params:", search.best_params_)
    else:
        clf = pipe
        clf.fit(X_train, y_train)

    # Predict and evaluate
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    roc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float("nan")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {"roc_auc": roc, "report": report, "estimator": clf}
    print(f"ROC-AUC: {roc:.4f}")

# -----------------------------
# Choose best by ROC-AUC
# -----------------------------
best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"] if not np.isnan(results[k]["roc_auc"]) else -1)
print(f"\nBest model selected: {best_name} (ROC-AUC: {results[best_name]['roc_auc']:.4f})")

best_clf = results[best_name]["estimator"]

# -----------------------------
# Save model and preprocessor
# -----------------------------
print("Saving model to:", MODEL_PATH)
joblib.dump(best_clf, MODEL_PATH)
# If you want the preprocessor separately, you can save it from pipeline
if hasattr(best_clf, 'named_steps') and 'preprocessor' in best_clf.named_steps:
    print("Saving preprocessor to:", PREPROCESSOR_PATH)
    joblib.dump(best_clf.named_steps['preprocessor'], PREPROCESSOR_PATH)

# -----------------------------
# Save training summary
# -----------------------------
summary = {
    "best_model": best_name,
    "roc_auc": float(results[best_name]["roc_auc"]),
}
with open("training_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print("Training complete. Artifacts:\n -", MODEL_PATH, "\n -", PREPROCESSOR_PATH, "\n - training_summary.json")