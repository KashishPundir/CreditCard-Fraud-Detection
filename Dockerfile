FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/final_fraud_xgboost_model.json models/
COPY models/final_fraud_scaler.pkl models/
COPY models/final_fraud_threshold.pkl models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]