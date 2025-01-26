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

def main():
    model = joblib.load(sys.argv[1]+"forest_best_model.pkl")

    step = normalize(sys.argv[2], min_step, max_step)
    type = normalize(transform_type(sys.argv[3]), min_type, max_type)
    amount = normalize(sys.argv[4], min_amount, max_amount)

    input_frame = pd.DataFrame({'step':[step], 'type':[type], 'amount':[amount]})
    predictions = model.predict(input_frame).tolist()

    print(json.dumps(predictions))

if __name__ == "__main__":
    main()

#python \load_model_command_line.py 470 CASH_IN 18000000