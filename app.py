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
    pod_id: str
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