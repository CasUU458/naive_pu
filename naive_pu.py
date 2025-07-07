import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

matplotlib.use("TkAgg")

import os
import pyarrow as pa

import torch
import time

# from classic_log_reg import ClassicLogReg 
# from datasets import DataSets
from classic_log_reg import ClassicLogReg
from datasets import DataSets
from naive_log_reg import NaiveLogReg

t = time.time()
print("Current time:", t)

# Load the Breast Cancer dataset
data = DataSets(name='BreastCancer')  # Use mock dataset for testing

c = 0.5  # Class frequency, can be adjusted

# Convert the dataset to PU dataset
X_train, y_train, X_test, y_test = data.get_X_y(test_size=0.5,
                                                c=c, labeling_mechanism="SCAR",
                                                train_balance=None,
                                                test_balance=None,
                                                scale_data=None,
                                                random_state=42)

print("Data loaded in {:.2f} seconds".format(time.time() - t))
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Fit the Classic Logistic Regression model
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

# Fit the Sklearn Logistic Regression model as a baseline
sk_clf = SklearnLogisticRegression(penalty=None, max_iter=100)
sk_clf.fit(X_train.values, y_train.values)
y_pred_sk = sk_clf.predict(X_test.values)
report_sk = classification_report(y_test.values, y_pred_sk, output_dict=True)
print("Sklearn Classification Report:")
print(pd.DataFrame(report_sk).transpose())
cm_sk = confusion_matrix(y_test.values, y_pred_sk)
print("Sklearn Confusion Matrix:")
print(cm_sk)

# Fit the Naive Logistic Regression model
naive_clf = NaiveLogReg()
naive_clf.fit(X_train.values, y_train.values)
y_pred_naive = naive_clf.predict(X_test.values)
report_naive = classification_report(y_test.values, y_pred_naive, output_dict=True)
print("Naive Logistic Regression Classification Report:")
print(pd.DataFrame(report_naive).transpose())
cm_naive = confusion_matrix(y_test.values, y_pred_naive)
print("Naive Confusion Matrix:")
print(cm_naive)


# Plot the loss curves for all models
def plot_loss_curves(classic_log_reg, naive_log_reg, c=None):
    plt.figure(figsize=(12, 6))
    plt.plot(classic_log_reg.loss_log, label='Classic Log Reg Loss', color='blue', alpha=0.5)
    plt.plot(naive_log_reg.loss_log, label='Naive Log Reg Loss', color='orange', alpha=0.5)
    plt.plot(naive_log_reg.loss_c_log, label='Naive Log Reg Loss C', color='green', alpha=0.5)
    plt.plot(naive_log_reg.c_log, label=r"$\hat{c}$", color='red', alpha=0.8, linestyle='--')
    if c is not None:
        plt.axhline(y=c, color='purple', linestyle='--', label='True c')
    plt.title('Loss Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig("loss_curves.png", bbox_inches='tight')


# plot_loss_curves(clf, naive_clf, c=c)

def plot_probabilities(clf, naive_clf, X_test, y_test):
    x_range = np.arange(X_test.shape[0])

    plt.figure()
    # Classic Logistic Regression probabilities
    y_prob_classic = clf.predict_proba(X_test.values)

    sorted_indices = np.argsort(y_prob_classic)
    sorted_probs_classic = y_prob_classic[sorted_indices]
    sorted_y_test = y_test.values[sorted_indices]
    correct_pred = np.abs(sorted_probs_classic - sorted_y_test) <= 0.5
    correct_pred = np.where(correct_pred, 'C2', 'C3')
    plt.scatter(x_range, sorted_probs_classic, c=correct_pred, alpha=0.5,label="Classic", marker="o", s=10)

    # Naive Logistic Regression probabilities
    y_prob_naive = naive_clf.predict_proba(X_test.values)
    sorted_indices_naive = np.argsort(y_prob_naive)
    sorted_probs_naive = y_prob_naive[sorted_indices_naive]
    sorted_y_test_naive = y_test.values[sorted_indices_naive]
    correct_pred_naive = np.abs(sorted_probs_naive - sorted_y_test_naive) <= 0.5
    correct_pred_naive = np.where(correct_pred_naive, 'C0', 'C1')
    plt.scatter(x_range, sorted_probs_naive, c=correct_pred_naive,label="Naive",marker="o", alpha=0.5, s=10)
    plt.title('Probabilities from Logistic Regression Models')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.xticks([])
    plt.savefig("probabilities.png", bbox_inches='tight')

plot_loss_curves(clf, naive_clf, c=c)
plot_probabilities(clf, naive_clf, X_test, y_test)
plt.show()
