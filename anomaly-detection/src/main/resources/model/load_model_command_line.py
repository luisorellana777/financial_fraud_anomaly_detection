import joblib
import sys
import json
import pandas as pd

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

def main():
    model = joblib.load("/Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/forest_best_model.pkl")

    step = normalize(sys.argv[1], min_step, max_step)
    type = normalize(transform_type(sys.argv[2]), min_type, max_type)
    amount = normalize(sys.argv[3], min_amount, max_amount)

    input_frame = pd.DataFrame({'step':[step], 'type':[type], 'amount':[amount]})
    predictions = model.predict(input_frame).tolist()

    print(json.dumps(predictions))

if __name__ == "__main__":
    main()

#python \load_model_command_line.py 470 CASH_IN 18000000