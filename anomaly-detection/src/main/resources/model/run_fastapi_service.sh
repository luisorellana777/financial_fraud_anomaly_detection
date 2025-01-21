#!/bin/bash
uvicorn --app-dir /Users/luisorellanaaltamirano/Documents/Machine_Learning/anomaly-detection/src/main/resources/model/ load_model_FastAPI:app --log-level info --port 8081