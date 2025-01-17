
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
import joblib
from scipy.stats import uniform, anderson
from sklearn import svm
from sklearn.svm import OneClassSVM, SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier as DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, make_scorer, f1_score

min_step = 1.000000;
max_step = 743.000000;
min_type = 0.000000;
max_type = 4.000000;
min_amount = 0.000000;
max_amount = 92445516.640000;

def normalize(value, min, max):
    return (float(value) - min) / (max - min);

def transform_type(type):
    match type:
        case 'CASH_IN':
            return 0
        case 'CASH_OUT':
            return 1
        case 'DEBIT':
            return 2
        case 'PAYMENT':
            return 3
        case 'TRANSFER':
            return 4

app = FastAPI()

import sklearn
print(sklearn.__version__)

model = joblib.load("/Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/forest_best_model.pkl")

class Features(BaseModel):
    step: float
    type: str
    amount: int

@app.post("/predict")
def predict(features: Features):
    try:
        input_data = [[
            normalize(features.step, min_step, max_step),
            normalize(transform_type(features.type), min_type, max_type),
            normalize(features.amount, min_amount, max_amount)
        ]]

        # Make prediction
        prediction = model.predict(input_data)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#uvicorn load_model_FastAPI:app --reload --port 8081
#curl -X POST "http://127.0.0.1:8081/predict" -H "Content-Type: application/json" -d '{"step": 350, "type": "CASH_OUT", "amount": 1800000}'