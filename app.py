
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# Load production bundle
# --------------------------------------------------
MODEL_PATH = "models/final_fraud_xgboost_model.json"
SCALER_PATH = "models/final_fraud_scaler.pkl"
THRESHOLD_PATH = "models/final_fraud_threshold.pkl"

from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("models/final_fraud_xgboost_model.json")

scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)

feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud prediction API using XGBoost",
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class Transaction(BaseModel):
    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CreditCardFraudXGBoost",
        "version": "1"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    try:

        # Convert request to DataFrame
        input_data = transaction.model_dump()

        X = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Same preprocessing used during training
        X["Amount"] = np.log1p(X["Amount"])

        # Same scaler used during training
        X_scaled = scaler.transform(X)

        # Fraud probability
        fraud_probability = float(
            model.predict_proba(X_scaled)[0, 1]
        )

        # Apply frozen threshold
        prediction = int(
            fraud_probability >= threshold
        )

        label = (
            "Fraud"
            if prediction == 1
            else "Legitimate"
        )

        return {
            "fraud_probability": fraud_probability,
            "threshold": threshold,
            "prediction": prediction,
            "prediction_label": label
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )