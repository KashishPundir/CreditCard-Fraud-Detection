# Credit Card Fraud Detection

An end-to-end machine learning system for detecting fraudulent credit card transactions using XGBoost, FastAPI, Docker, Docker Hub, and GitHub.

## Project Overview

This project follows a production-oriented machine learning workflow:

- Time-based train/validation/test split
- Data preprocessing and feature scaling
- Multiple ML model comparison
- XGBoost model tuning
- Precision-Recall AUC evaluation
- Fraud detection threshold optimization
- Final model evaluation on an untouched test set
- MLflow experiment tracking
- FastAPI inference API
- Docker containerization
- Docker Hub image publishing
- Git/GitHub version control

## Technology Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Docker Hub
- MLflow
- Git
- GitHub
  
## Model

The final production model is an XGBoost classifier.

The fraud decision threshold was selected on the validation set using the project's operational objective:

> Maximize fraud recall subject to a maximum acceptable number of false positives.

Final validation threshold:

```text
0.09133733808994293
```

<img width="270" height="204" alt="image" src="https://github.com/user-attachments/assets/952c8e62-45af-4015-b567-25a96fe5db9a" />

