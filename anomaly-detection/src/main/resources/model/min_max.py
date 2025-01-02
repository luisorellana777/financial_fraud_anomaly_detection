#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:17:39 2024

@author: luisorellanaaltamirano
"""

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


df = pd.read_csv("/Users/luisorellanaaltamirano/Documents/Machine_Learning/Synthetic_Financial_datasets_log.csv")

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

print("Step min %f"%df.step.min())
print("Step max %f"%df.step.max())
print("Type min %f"%df.type.min())
print("Type max %f"%df.type.max())
print("Amount min %f"%df.amount.min())
print("Amount max %f"%df.amount.max())


df = (df - df.min()) / (df.max() - df.min())

print("-----------------------------------------")

print("Step min %f"%df.step.min())
print("Step max %f"%df.step.max())
print("Type min %f"%df.type.min())
print("Type max %f"%df.type.max())
print("Amount min %f"%df.amount.min())
print("Amount max %f"%df.amount.max())