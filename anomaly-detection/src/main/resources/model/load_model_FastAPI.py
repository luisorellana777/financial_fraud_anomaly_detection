import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

min_step = 1.000000;
max_step = 743.000000;
min_type = 0.000000;
max_type = 4.000000;
min_amount = 0.000000;
max_amount = 92445516.640000;

def normalize(value, min, max):
    return (float(value) - min) / (max - min);

def transform_type(type):
    if type == 'CASH_IN':
        return 0
    elif type == 'CASH_OUT':
        return 1
    elif type == 'DEBIT':
        return 2
    elif type == 'PAYMENT':
        return 3
    elif type == 'TRANSFER':
        return 4

app = FastAPI()

model_path = os.getenv("MODEL_PATH")
model = joblib.load(model_path+"forest_best_model.pkl")

class Features(BaseModel):
    step: float
    type: str
    amount: int

@app.post("/predict")
def predict(features: Features):
    try:

        input_frame = pd.DataFrame({'step':[normalize(features.step, min_step, max_step)],
                                    'type':[normalize(transform_type(features.type), min_type, max_type)],
                                    'amount':[normalize(features.amount, min_amount, max_amount)]})

        # Make prediction
        prediction = model.predict(input_frame)
        return prediction.tolist()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#uvicorn load_model_FastAPI:app --reload --port 8081
#curl -X POST "http://127.0.0.1:8081/predict" -H "Content-Type: application/json" -d '{"step": 350, "type": "CASH_OUT", "amount": 1800000}'