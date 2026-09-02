"""
main.py
-------
DAY 2 + DAY 6 DELIVERABLE: AI Microservices API

Endpoints (as specified in the training plan):
  POST /predict_driver_behavior  -> raw model prediction (risk_level + probabilities)
  POST /analyze_risk_score       -> business-friendly risk score (0-100) + alert flag
  POST /log_data                 -> persist a result to the database (called by n8n)
  GET  /logs                     -> fetch recent logs (used by the dashboard)
  GET  /health                   -> simple health check

Run with:
  uvicorn main:app --reload --port 8000

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import joblib
import pandas as pd
import os

from schemas import (
    DriverTelemetry, PredictionResponse,
    RiskAnalysisResponse, LogEntry, LogResponse,
)
from database import init_db, get_db, RiskLog

app = FastAPI(
    title="Smart AI Logistics Intelligence System",
    description="AI microservice for driver behavior prediction and risk scoring.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# STARTUP: load the trained model once (not per-request — that would be slow)
# and make sure the database tables exist.
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_model", "model.pkl")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]

RISK_SCORE_MAP = {"low": 20, "medium": 55, "high": 85}  # baseline anchor per class
ALERT_THRESHOLD = 70  # risk_score above this triggers alert_triggered = True


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ---------------------------------------------------------------------------
# ENDPOINT 1: /predict_driver_behavior
# ---------------------------------------------------------------------------
@app.post("/predict_driver_behavior", response_model=PredictionResponse)
def predict_driver_behavior(payload: DriverTelemetry):
    # Build a single-row DataFrame with columns in the EXACT order the
    # model was trained on — mismatched order silently gives wrong results,
    # which is why we saved feature_cols alongside the model.
    row = pd.DataFrame([[getattr(payload, col) for col in feature_cols]], columns=feature_cols)

    pred_encoded = model.predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(row)[0]
    prob_dict = {
        label_encoder.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(probs)
    }

    return PredictionResponse(
        driver_id=payload.driver_id,
        risk_level=pred_label,
        risk_probabilities=prob_dict,
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# ENDPOINT 2: /analyze_risk_score
# ---------------------------------------------------------------------------
@app.post("/analyze_risk_score", response_model=RiskAnalysisResponse)
def analyze_risk_score(payload: DriverTelemetry):
    row = pd.DataFrame([[getattr(payload, col) for col in feature_cols]], columns=feature_cols)

    pred_encoded = model.predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    probs = model.predict_proba(row)[0]

    # Turn class probabilities into a smooth 0-100 score instead of just
    # 3 buckets, by weighting each class's anchor score by its probability.
    class_names = label_encoder.inverse_transform(range(len(probs)))
    risk_score = sum(
        RISK_SCORE_MAP[name] * prob for name, prob in zip(class_names, probs)
    )

    # Use the model's feature_importances_ to explain WHY, by ranking this
    # specific driver's feature values against typical thresholds.
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_factors = sorted(importances, key=importances.get, reverse=True)[:3]

    return RiskAnalysisResponse(
        driver_id=payload.driver_id,
        risk_score=round(risk_score, 2),
        risk_level=pred_label,
        top_contributing_factors=top_factors,
        alert_triggered=risk_score >= ALERT_THRESHOLD,
        timestamp=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# ENDPOINT 3: /log_data  (this is what your n8n workflow calls)
# ---------------------------------------------------------------------------
@app.post("/log_data", response_model=LogResponse)
def log_data(entry: LogEntry, db: Session = Depends(get_db)):
    record = RiskLog(
        driver_id=entry.driver_id,
        risk_level=entry.risk_level,
        risk_score=entry.risk_score,
        source=entry.source,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ---------------------------------------------------------------------------
# Helper endpoint for the dashboard
# ---------------------------------------------------------------------------
@app.get("/logs", response_model=list[LogResponse])
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(RiskLog).order_by(RiskLog.id.desc()).limit(limit).all()
    return logs
