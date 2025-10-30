# Next Pod Failure Prediction — Project Files

This document contains a complete, ready-to-run project: training, evaluation, and a FastAPI deployment for predicting next pod failures using your synthetic dataset. Copy the files into a project folder and follow the README instructions at the end.

---

## File: `train_model.py`
```python
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
```

---

## File: `evaluate_model.py`
```python
"""
evaluate_model.py — Load trained model and run evaluation + single/batch predictions
"""
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score

MODEL_PATH = "pod_failure_predictor.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH + " not found. Train the model first.")

model = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# If dataset available, evaluate on it
if os.path.exists("pod_failure_prediction_dataset.csv"):
    df = pd.read_csv("pod_failure_prediction_dataset.csv")
    target_col = "predicted_pod_failure" if "predicted_pod_failure" in df.columns else "next_pod_failure"
    X = df.drop(columns=[target_col, "timestamp"]) if "timestamp" in df.columns else df.drop(columns=[target_col])
    y = df[target_col]
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    print("ROC-AUC:", roc_auc_score(y, y_proba))
    print(classification_report(y, y_pred))
else:
    print("No dataset found for full evaluation. Provide a CSV named pod_failure_prediction_dataset.csv to run evaluation.")

# Single prediction example
example = {
    "cpu_usage_pct": 88.5,
    "memory_usage_pct": 95.2,
    "memory_leak_rate": 0.22,
    "restart_count_24h": 3,
    "error_log_rate": 10,
    "request_latency_ms": 180,
    "replica_count": 4,
    "node_pressure_score": 0.74,
    "autoscaler_action": "scale_up",
    "prometheus_anomaly_score": 0.82,
    "previous_failures": 2,
    "deployment_uptime_hrs": 48,
}

input_df = pd.DataFrame([example])
prob = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]
print("Example prediction -> probability:", prob, "pred:", pred)
```

---

## File: `app.py` (FastAPI service)
```python
"""
app.py — FastAPI app to serve pod failure predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = "pod_failure_predictor.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(MODEL_PATH + " not found. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Self-Healing K8s Pod Failure Predictor API")

class PodMetrics(BaseModel):
    cpu_usage_pct: float
    memory_usage_pct: float
    memory_leak_rate: float
    restart_count_24h: int
    error_log_rate: int
    request_latency_ms: float
    replica_count: int
    node_pressure_score: float
    autoscaler_action: str
    prometheus_anomaly_score: float
    previous_failures: int
    deployment_uptime_hrs: int

@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}

@app.post("/predict")
def predict(metrics: PodMetrics):
    input_df = pd.DataFrame([metrics.dict()])
    try:
        proba = model.predict_proba(input_df)[0][1]
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "input": metrics.dict(),
        "failure_probability": round(float(proba), 4),
        "prediction": "at_risk" if pred == 1 else "stable"
    }

# To run locally:
# uvicorn app:app --reload --port 8000
```

---

## File: `requirements.txt`
```
fastapi==0.95.2
uvicorn==0.22.0
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
joblib
matplotlib
seaborn
```

---

## File: `Dockerfile`
```dockerfile
# Use official python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## File: `docker-compose.yml` (optional)
```yaml
version: "3.8"
services:
  predictor:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./:/app
    environment:
      - ENV=production
```

---

## File: `README.md`
```markdown
# Next Pod Failure Prediction

This project trains a binary classifier to predict the next pod failure in a Kubernetes cluster and serves it via FastAPI.

## Files
- `train_model.py` — training pipeline; produces `pod_failure_predictor.pkl`
- `evaluate_model.py` — evaluation + single example prediction
- `app.py` — FastAPI service exposing `/predict`
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`

## Quickstart
1. Place `pod_failure_prediction_dataset.csv` in the project root.
2. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Train the model:

```bash
python train_model.py
```

4. Run the API locally:

```bash
uvicorn app:app --reload --port 8000
```

5. Predict (example):

```bash
curl -X POST 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' -d @example.json
```

## Notes
- The training script will detect common target column names (`predicted_pod_failure`, `next_pod_failure`).
- Save and version the `pod_failure_predictor.pkl` artifact and consider CI/CD for retraining on new metrics.
```

---

### Next steps
- Run `python train_model.py` to create the model artifact.
- Use `uvicorn app:app --reload` to serve predictions locally.
- If you want, I can: generate a `Kubernetes` manifest for deployment, add Prometheus scraping, or wire this into your MCP pipeline for automated live predictions.

