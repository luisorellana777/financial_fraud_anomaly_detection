import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
import joblib
from scipy.stats import uniform
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import OneClassSVM, SVC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix, make_scorer, f1_score


def calculate_chisquare_print(independent_variable, dependent_variable):
  ct=pd.crosstab(independent_variable, dependent_variable)
  cs= scipy.stats.chi2_contingency(ct) 
  print ('pValue: %f'%cs[1])

def calculate_logisticregression_print(independent_variable, dependent_variable):
  independent_variable = sm.add_constant(independent_variable) 
  model = sm.Logit(dependent_variable, independent_variable).fit()
  print(model.summary())
  


df_original = pd.read_csv("/Users/luisorellanaaltamirano/Documents/Machine_Learning/Synthetic_Financial_datasets_log.csv")

# Description of the data
print(df_original.columns)

df_original['isFraud'] = df_original['isFraud'] | df_original['isFlaggedFraud']
df_original.drop(['isFlaggedFraud'], inplace=True, axis=1)

df = df_original.copy()

#DATA ANALISIS

fraud_amount = df_original[df_original['isFraud'] == 1]['amount'].mean()
non_fraud_amount = df_original[df_original['isFraud'] == 0]['amount'].mean()

all_amount = [fraud_amount, non_fraud_amount]

counts = df_original['isFraud'].value_counts()
avg_amounts = df_original.groupby('isFraud')['amount'].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First subplot: Number of Transactions
bars1 = axes[0].bar(
  ['Non-Fraud', 'Fraud'],
  counts,
  color=['skyblue', 'orange'],
  alpha=0.8,
  edgecolor='black'
)
axes[0].set_title('Number of Transactions')
axes[0].set_ylabel('Count')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# Adding counts and percentages
total_count = counts.sum()
for bar, count in zip(bars1, counts):
  yval = bar.get_height()
  percentage = (count / total_count) * 100
  axes[0].text(
    bar.get_x() + bar.get_width() / 2,
    yval,
    f'({percentage:.1f}%)',
    ha='center',
    va='bottom'
  )

# Second subplot: Average Transaction Amount
bars2 = axes[1].bar(
  ['Non-Fraud', 'Fraud'],
  avg_amounts,
  color=['skyblue', 'orange'],
  alpha=0.8,
  edgecolor='black'
)
axes[1].set_title('Average Transaction Amount')
axes[1].set_ylabel('Average Amount')
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adding average amounts
for bar in bars2:
  yval = bar.get_height()
  axes[1].text(
    bar.get_x() + bar.get_width() / 2,
    yval,
    f'{yval:.2f}',
    ha='center',
    va='bottom'
  )

plt.tight_layout()
plt.show()

###################
counts = df_original['isFraud'].value_counts()
plt.figure(figsize = (6,6))
plt.pie(counts, labels = counts.index, autopct = "%1.1f%%", colors=['skyblue', 'orange', 'red', 'yellow', 'green'], shadow = True,explode = (0, 0),textprops={'fontsize': 15})
plt.title('Count of each isFraud of transaction', fontweight = 'bold', fontsize = 18, fontfamily = 'times new roman')
plt.show()


#convert categorical data to integers
le = LabelEncoder()
df['nameOrig'] = le.fit_transform(df['nameOrig'])
df['nameDest'] = le.fit_transform(df['nameDest'])
df['type'] = le.fit_transform(df['type']) # 0.0=CASH_IN; 0.25=CASH_OUT; 0.5=DEBIT; 0.75=PAYMENT; 1.0=TRANSFER


# pearson corrleation matrix of the numerical data
correlation = df.corr()
# visulaising the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation,vmin = -1, vmax = 1,cmap = "Greys",annot = True, fmt = '.2f')
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.xticks(rotation = 45)
plt.show()

#Global Type
counts = df_original.groupby('type').count()['amount']
plt.figure(figsize = (6,6))
plt.pie(counts, labels = counts.index, autopct = "%1.1f%%", colors=['skyblue', 'orange', 'red', 'yellow', 'green'], shadow = True,explode = (0.1, 0, 0, 0, 0),textprops={'fontsize': 15})
plt.title('Count of each type of transaction', fontweight = 'bold', fontsize = 18, fontfamily = 'times new roman')
plt.show()

#Types which are fraud
counts = df_original[(df_original['isFraud']==1)].groupby('type').count()['amount']
plt.figure(figsize = (6,6))
plt.pie(counts, labels = counts.index, autopct = "%1.1f%%", colors=['skyblue', 'orange', 'red', 'yellow', 'green'], shadow = True,explode = (0, 0),textprops={'fontsize': 15})
plt.title('Count of each type of fraud transaction', fontweight = 'bold', fontsize = 18, fontfamily = 'times new roman')
plt.show()

########### Hypothesis Testing #################
'''
#Does step have any relationship with fraudulent transaction?
# Significant pValue. Not eliminate it.
calculate_logisticregression_print(df_original['step'] , df_original['isFraud'])

#Does type have any relationship with fraudulent transaction?
# Significant pValue. Not eliminate it.
calculate_chisquare_print(df_original['type'] , df_original['isFraud'])

#Does amount have any relationship with fraudulent transaction?
# Significant pValue. Not eliminate it.
calculate_logisticregression_print(df_original['amount'] , df_original['isFraud'])

#Does nameOrig have any relationship with fraudulent transaction?
# Not significant pValue. Eliminate it.
calculate_logisticregression_print(df['nameOrig'] , df['isFraud'])

#Does oldbalanceOrg have any relationship with fraudulent transaction?
# Significant pValue. Eliminate it though, since documentations recomends it.
calculate_logisticregression_print(df_original['oldbalanceOrg'] , df_original['isFraud'])

#Does newbalanceOrig have any relationship with fraudulent transaction?
# Significant pValue. Eliminate it though, since documentations recomends it.
calculate_logisticregression_print(df_original['newbalanceOrig'] , df_original['isFraud'])

#Does nameDest have any relationship with fraudulent transaction?
# Significant pValue. Not eliminate it.
calculate_chisquare_print(df['nameDest'] , df_original['isFraud'])

#Does oldbalanceDest have any relationship with fraudulent transaction?
# Significant pValue. Eliminate it though, since documentations recomends it.
calculate_logisticregression_print(df_original['oldbalanceDest'] , df_original['isFraud'])

#Does newbalanceDest have any relationship with fraudulent transaction?
# Not significant pValue. Eliminate it.
calculate_logisticregression_print(df_original['newbalanceDest'] , df_original['isFraud'])
'''

########### Data Preparation #################


# Removing Unessesary fields
df.drop(['nameOrig'], inplace=True, axis=1)
df.drop(['oldbalanceOrg'], inplace=True, axis=1)
df.drop(['newbalanceOrig'], inplace=True, axis=1)
df.drop(['oldbalanceDest'], inplace=True, axis=1)
df.drop(['newbalanceDest'], inplace=True, axis=1)
df.drop(['nameDest'], inplace=True, axis=1)
isFraud_field_no_normalise = df['isFraud']
df = df.drop('isFraud', axis = 1)

# normalize numbers in dataframe before joining one-hot encoded values
df = (df - df.min()) / (df.max() - df.min())

# separating feature variables and class variables
X = df
y = isFraud_field_no_normalise

# splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

########### Predictive Models #################

########### Logistic Regression
'''
logreg = LogisticRegression()

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)
#Best hyperparameters: {'C': 0.01, 'solver': 'liblinear'}

# Print the best score achieved by GridSearchCV
print("Best score:", grid_search.best_score_)
#Best score: 0.9987026907605161

# Make predictions on the test data using the best model
y_pred = grid_search.best_estimator_.predict(X_test)

# recall of the logistic regression
recall_lr = recall_score(y_test, y_pred)

# classification report
classification_lr = classification_report(y_test, y_pred)

print(f"Recall of logistic regression {recall_lr}")
print(f"Classification Report of logistic regression\n {classification_lr}")
'''
########### Support Vecto Machine (SVM)
#No Grid Search

# Adjust class weights: make anomalies more important (heavier weight for class 1)
class_weights = {0: 1, 1: 10}  # 0 for normal, 1 for anomalies

# Fit the SVM classifier with class weights
clf = svm.SVC(kernel="rbf", gamma="scale", C=50)
clf.fit(X_train, y_train)

# Predict the labels (0 for normal, 1 for anomalous)
y_pred = clf.predict(X_test)

joblib.dump(clf, 'svm_model.pkl')

#Mesure model
recall_lr = recall_score(y_test, y_pred)
# classification report
classification_lr = classification_report(y_test, y_pred)
print(f"Recall of SVC {recall_lr}")
print(f"Classification Report of SVC\n {classification_lr}")


#Grid Search
'''
svc = SVC()

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 10, 100],
    'gamma': ['scale', 0.1, 10],
    'kernel': ['poly', 'rbf'],
    'class_weight': [{0: 1, 1: 10}, {0: 1, 1: 100}]  # Automatically handle class imbalance
}

# Perform grid search
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='f1', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model
print("Best Parameters:", grid_search.best_params_)

# Evaluate on the test set
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'svm_best_model.pkl')
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Mesure model
recall_lr = recall_score(y_test, y_pred)
# classification report
classification_lr = classification_report(y_test, y_pred)
print(f"Recall of SVC {recall_lr}")
print(f"Classification Report of SVC\n {classification_lr}")
'''

'''
########### Support Vecto Machine (One-Class SVM)
# Define the parameter distributions
param_distributions = {
    "nu": uniform(0.01, 0.3),  # Continuous range for the anomaly proportion
    "kernel": ["rbf", "poly", "sigmoid"],  # Kernel types
    "gamma": uniform(0.001, 1)  # Continuous range for kernel coefficient
}

# Custom scorer based on F1 score
def f1_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_pred = np.where(y_pred == 1, 1, -1)  # Map predictions back to labels
    return f1_score(y, y_pred, pos_label=-1)  # Anomalies are the positive class

scorer = make_scorer(f1_scorer)

# Perform randomized search
svc = OneClassSVM()
random_search = RandomizedSearchCV(
    svc,
    param_distributions,
    n_iter=20,  # Number of parameter combinations to try
    scoring=scorer,
    cv=3,  # 3-fold cross-validation
    random_state=42,
    verbose=2,
    n_jobs=-1  # Use all available CPUs
)
random_search.fit(X_train, y_train)

# Print best parameters and score
print("Best Parameters:", random_search.best_params_)
print("Best F1 Score (CV):", random_search.best_score_)

# Evaluate on test data
best_model = random_search.best_estimator_
joblib.dump(best_model, 'svm_one_class_model.pkl')

y_test_pred = best_model.predict(X_test)
y_test_pred = np.where(y_test_pred == 1, 1, -1)
f1 = f1_score(y_test, y_test_pred, pos_label=-1)
print("Test F1 Score:", f1)
'''
