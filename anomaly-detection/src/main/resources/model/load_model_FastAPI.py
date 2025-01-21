import joblib
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

model = joblib.load("/Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/forest_best_model.pkl")

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
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#uvicorn load_model_FastAPI:app --reload --port 8081
#curl -X POST "http://127.0.0.1:8081/predict" -H "Content-Type: application/json" -d '{"step": 350, "type": "CASH_OUT", "amount": 1800000}'