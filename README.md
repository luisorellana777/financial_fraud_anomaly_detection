# Financial Fraud Anomaly Detection
This project aims to detect fraudulent financial transactions using machine learning and anomaly detection techniques. The repository includes data preprocessing process, exploratory data analysis (EDA), and implementations of algorithms to identify anomalies in transaction patterns.

Table of Contents
- Features
- Dataset
- Usage
- Models and Techniques
- Results
- Project Structure
- Contributing
- Contact


## Features
- Exploratory Data Analysis to understand transaction patterns and identify trends.
- Data preprocessing to handle missing data, outliers, and imbalanced classes.
- Implementation of anomaly detection algorithms including:
    - Logistic Regression
    - Support Vector Machine
- Evaluation metrics to measure the effectiveness of fraud detection.
- Visualizations to interpret results and gain insights.


## Dataset
The project uses a dataset of anonymized financial transactions labeled as fraudulent or non-fraudulent.

- Source: [A financial mobile money simulator for fraud detection](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf)
- Structure:
  - Features represent transaction details (e.g., amount, time, frequency).
  - Labels indicate whether the transaction is fraudulent (1) or not (0).

Preprocessing Steps:

- Handled missing or inconsistent values.
- Scaled numerical features for model compatibility.
- Addressed data imbalance using techniques like SMOTE or undersampling.


## Usage
Chose between [FastAPI Integration](https://github.com/luisorellana777/financial_fraud_anomaly_detection/blob/master/anomaly-detection/src/main/resources/model/load_model_FastAPI.py), or [Python Script Integration](https://github.com/luisorellana777/financial_fraud_anomaly_detection/blob/master/anomaly-detection/src/main/resources/model/load_model_command_line.py) through this [property](https://github.com/luisorellana777/financial_fraud_anomaly_detection/blob/master/anomaly-detection/src/main/resources/application.yml#L8).


Run in [Spring Boot service folder](https://github.com/luisorellana777/financial_fraud_anomaly_detection/tree/master/anomaly-detection):
```bash
./gradlew bootRun
```


## Models and Techniques
This project uses the following algorithms for anomaly detection:

Logistic Regression: Detects anomalies by hyperplane of separation.
Support Vector Machine: Identifies patterns using support vector methods.

These methods are suitable for imbalanced datasets and require explicit labels for fraud cases.

The results of the project are summarized below:

Performance Metrics:

- Precision: 79%
- Recall: 60%
- F1-Score: 62%

Find detailed results in [this directory](https://github.com/luisorellana777/financial_fraud_anomaly_detection/blob/master/Anomaly%20Detection.ipynb).


## Project Structure
```bash
financial_fraud_anomaly_detection/
├── anomaly_detection/                         # Spring Boot Service
├── src/                                       # Source code for preprocessing and model training
│   └── main                             
│       └── resources        
│           └── model       
│               └── load_model_FastAPI.py      # Python FastAPI controller service to read model
│               ├── load_model_command_line.py # Python script to read model
│               ├── run_fastapi_service.sh     # Script to run FastAPI service.
│               └── svm_model.pkl              # SVM saved model used by service.
├── Anomaly Detection.ipynb                    # Jupyter Notebooks for EDA and experiments
├── Anomaly Detection Script.py                # Script experiments
├── Data Science - Anomaly Detection.pdf       # Talk slides
├── SVM_CUDA_Training.ipynb.                   # Jupyter Notebooks for SVM with NVidia CUDA
├── anomaly_detection.postman_collection.json  # postman request for service to predict trx.
├── svm_model.pkl                              # Saved model after running python script.
└── README.md                                  # Project documentation
```


## Contributing
Contributions are welcome!

1. Fork the repository.
2. Create a feature branch:
```bash
git checkout -b feature-name
```
3. Commit your changes and push to your branch.
4. Submit a pull request.


## 🔗 Contact
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/luis-maximo-orellana-altamirano)