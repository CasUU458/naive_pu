import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification,load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import os
import pyarrow  as pa

import torch
import time

# from classic_log_reg import ClassicLogReg 
# from datasets import DataSets
from classic_log_reg import ClassicLogReg
from datasets import DataSets   









t = time.time()
print("Current time:", t)


# Load the Breast Cancer dataset 
data = DataSets(name='BreastCancer')

#Convert the dataset to PU dataset
X_train, y_train, X_test, y_test = data.get_X_y(test_size=0.99,
    c=1,labeling_mechanism="SCAR",
    train_balance=None,
    test_balance=None,
    scale_data="standard",
    random_state=42)
print("Data loaded in {:.2f} seconds".format(time.time() - t)   )
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

#Fit the Classic Logistic Regression model
clf = ClassicLogReg()
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
report = classification_report(y_test.values, y_pred, output_dict=True)
print("training completed in {:.2f} seconds".format(time.time() - t))
print("Classification Report:")
print(pd.DataFrame(report).transpose())
cm = confusion_matrix(y_test.values, y_pred)    
print("Confusion Matrix:")
print(cm)

#Fit the Sklearn Logistic Regression model as a baseline
sk_clf = SklearnLogisticRegression(penalty=None,max_iter=1000)
sk_clf.fit(X_train.values, y_train.values)
y_pred_sk = sk_clf.predict(X_test.values)
report_sk = classification_report(y_test.values, y_pred_sk, output_dict=True)
print("Sklearn Classification Report:")
print(pd.DataFrame(report_sk).transpose())
cm_sk = confusion_matrix(y_test.values, y_pred_sk)    
print("Sklearn Confusion Matrix:")  
print(cm_sk)