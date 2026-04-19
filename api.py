"""
FastAPI Service — Revenue Forecast API
=======================================
Serves the trained model as a REST API.
Endpoints:
  GET  /health          — Health check
  GET  /forecast        — Predict next month's revenue (from saved model)
  POST /forecast/custom — Predict from custom input features

Usage:
  uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Revenue Forecasting API",
    description="Predicts next month's revenue using Linear Regression on historical billing data.",
    version="1.0.0"
)

# ── Load model on startup ──────────────────────────────────────
MODEL_PATH = "outputs/model.pkl"
model_data = None

def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}. Run main.py first.")

load_model()


# ── Request / Response Schemas ─────────────────────────────────
class ForecastRequest(BaseModel):
    month: int                          # 1-12
    quarter: int                        # 1-4
    revenue_lag_1: float                # Last month's revenue
    revenue_lag_2: float                # 2 months ago
    revenue_lag_3: float                # 3 months ago
    rolling_mean_3: float               # 3-month avg
    rolling_mean_6: float               # 6-month avg
    rolling_std_3: float                # 3-month std dev
    revenue_growth: float               # MoM growth rate
    invoices: Optional[float] = 0.0    # Optional extra feature
    customers: Optional[float] = 0.0
    avg_deal_size: Optional[float] = 0.0

class ForecastResponse(BaseModel):
    predicted_revenue: float
    model_type: str
    currency: str = "USD"
    note: str


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_data is not None,
        "model_path": MODEL_PATH
    }


@app.post("/forecast/custom", response_model=ForecastResponse)
def forecast_custom(req: ForecastRequest):
    """Predict revenue from custom input features."""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run main.py first.")

    model   = model_data["model"]
    scaler  = model_data["scaler"]
    features = model_data["features"]

    # Build feature dict
    input_dict = {
        "month"          : req.month,
        "quarter"        : req.quarter,
        "month_sin"      : np.sin(2 * np.pi * req.month / 12),
        "month_cos"      : np.cos(2 * np.pi * req.month / 12),
        "is_q4"          : int(req.month in [10, 11, 12]),
        "revenue_lag_1"  : req.revenue_lag_1,
        "revenue_lag_2"  : req.revenue_lag_2,
        "revenue_lag_3"  : req.revenue_lag_3,
        "rolling_mean_3" : req.rolling_mean_3,
        "rolling_mean_6" : req.rolling_mean_6,
        "rolling_std_3"  : req.rolling_std_3,
        "revenue_growth" : req.revenue_growth,
        "invoices"       : req.invoices,
        "customers"      : req.customers,
        "avg_deal_size"  : req.avg_deal_size,
    }

    # Only use features the model was trained on
    X = pd.DataFrame([{k: input_dict.get(k, 0) for k in features}])
    X_sc = scaler.transform(X)
    prediction = model.predict(X_sc)[0]

    return ForecastResponse(
        predicted_revenue=round(float(prediction), 2),
        model_type=type(model).__name__,
        note="Prediction based on provided features using trained Linear Regression model."
    )


@app.get("/")
def root():
    return {
        "message": "Revenue Forecasting API",
        "docs": "/docs",
        "endpoints": ["/health", "/forecast/custom"]
    }
