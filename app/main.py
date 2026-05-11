"""
main.py — Churn Inference API (FastAPI)

Endpoints:
  GET  /health       — API status and model version
  POST /predict      — prediction for a single customer
  POST /predict/batch — prediction for multiple customers

Start:
  cd /Users/Shaiane/Projeto_MLOPs_Bank_Churn-English
  source venv/bin/activate
  uvicorn app.main:app --reload --port 8000
"""
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
for _p in [str(ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.inference import ChurnPredictor

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = 'Bank Churn Prediction API',
    description = 'Predicts the probability of bank customer churn.',
    version     = '1.0.0',
)

# Load the model once at startup
_predictor: ChurnPredictor | None = None

@app.on_event('startup')
def load_model():
    global _predictor
    _predictor = ChurnPredictor()


# ── Schemas ──────────────────────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    CreditScore:      float = Field(..., example=650)
    Age:              float = Field(..., example=42)
    Tenure:           float = Field(..., example=5)
    Balance:          float = Field(..., example=125000.0)
    NumOfProducts:    float = Field(..., example=2)
    HasCrCard:        float = Field(..., example=1)
    IsActiveMember:   float = Field(..., example=1)
    EstimatedSalary:  float = Field(..., example=80000.0)
    Geography_France: float = Field(0, example=1)
    Geography_Germany:float = Field(0, example=0)
    Geography_Spain:  float = Field(0, example=0)
    Gender_Female:    float = Field(0, example=0)
    Gender_Male:      float = Field(1, example=1)
    AgeGroup_Middle:  float = Field(0, example=1)
    AgeGroup_Senior:  float = Field(0, example=0)
    AgeGroup_Young:   float = Field(0, example=0)
    AgeInactivity:        float = Field(0, example=0)
    EngagementScore:      float = Field(0, example=2)
    BalanceSalaryRatio:   float = Field(0, example=1.56)
    ProductsPerYear:      float = Field(0, example=0.4)

    model_config = {'extra': 'allow'}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_flag:        int
    risk_level:        str
    threshold_used:    float
    model_source:      str


class BatchRequest(BaseModel):
    customers: list[dict[str, Any]]
    threshold: float | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    if _predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    return {
        'status':       'ok',
        'model_source': _predictor.source,
        'threshold':    _predictor.threshold,
    }


@app.post('/predict', response_model=PredictionResponse)
def predict(customer: CustomerFeatures, threshold: float | None = None):
    if _predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    try:
        result = _predictor.predict_single(customer.model_dump(), threshold=threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post('/predict/batch')
def predict_batch(request: BatchRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail='Model not loaded.')
    if not request.customers:
        raise HTTPException(status_code=400, detail='Customer list is empty.')
    try:
        import pandas as pd
        t0  = time.time()
        df  = pd.DataFrame(request.customers)
        out = _predictor.predict(df, threshold=request.threshold)
        elapsed = round(time.time() - t0, 4)
        return {
            'n_customers':  len(df),
            'n_churn_flag': int(out['churn_flag'].sum()),
            'latency_s':    elapsed,
            'predictions':  out.to_dict(orient='records'),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
