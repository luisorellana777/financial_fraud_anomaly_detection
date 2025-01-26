#!/bin/bash
uvicorn --app-dir $1 load_model_FastAPI:app --reload --log-level info --port 8081