"""
schemas.py
----------
Pydantic models = the "contract" for your API. They do two jobs:
1. VALIDATION: if a client sends a string where a number is expected,
   FastAPI rejects the request automatically (422 error) before your
   code even runs — you don't need manual if-checks.
2. DOCUMENTATION: FastAPI reads these classes to auto-generate the
   Swagger UI at /docs, so your API documents itself.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DriverTelemetry(BaseModel):
    """Input shape for /predict_driver_behavior and /analyze_risk_score."""
    driver_id: str = Field(..., example="DRV-1001")
    avg_speed_kmh: float = Field(..., ge=0, le=200)
    speeding_events: int = Field(..., ge=0)
    harsh_braking_events: int = Field(..., ge=0)
    harsh_acceleration_events: int = Field(..., ge=0)
    sharp_turns: int = Field(..., ge=0)
    driving_hours: float = Field(..., ge=0, le=24)
    night_driving_pct: float = Field(..., ge=0, le=1)
    phone_usage_events: int = Field(..., ge=0)
    fatigue_score: float = Field(..., ge=0, le=1)


class PredictionResponse(BaseModel):
    driver_id: str
    risk_level: str          # "low" | "medium" | "high"
    risk_probabilities: dict # e.g. {"low": 0.7, "medium": 0.2, "high": 0.1}
    timestamp: datetime


class RiskAnalysisResponse(BaseModel):
    driver_id: str
    risk_score: float        # 0-100 numeric score derived from the model
    risk_level: str
    top_contributing_factors: list[str]
    alert_triggered: bool
    timestamp: datetime


class LogEntry(BaseModel):
    """Used by /log_data — this is what n8n will POST after each prediction
    so results get persisted to the database."""
    driver_id: str
    risk_level: str
    risk_score: float
    source: Optional[str] = "n8n"


class LogResponse(BaseModel):
    id: int
    driver_id: str
    risk_level: str
    risk_score: float
    timestamp: datetime
