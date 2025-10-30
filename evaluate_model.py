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
    "pod_id": "pod-test-1",
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