import pickle
import numpy as np
import sys
import json
import pandas as pd

def main():
    with open("/Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    input_frame = pd.DataFrame({'step':[sys.argv[1]], 'type':[sys.argv[2]], 'amount':[sys.argv[3]]})
    predictions = model.predict(input_frame).tolist()
    #python \load_model.py 1.0 0.25 0.003674

    print(json.dumps(predictions))  # Output as JSON string

if __name__ == "__main__":
    main()