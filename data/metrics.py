import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import logging
from config import CONFIG

def do_classification(classifier, name, X_train, y_train, X_test, y_test):
    t = time.time()
    print(f"starting classification for {name}")
    
    classifier.fit(X_train.values, y_train.values)
    y_pred = classifier.predict(X_test.values)
    report = classification_report(y_test.values, y_pred, output_dict=True)
    print("training completed in {:.2f} seconds".format(time.time() - t))
    print(f"{name} - Classification Report:")
    print(pd.DataFrame(report).transpose())
    cm = confusion_matrix(y_test.values, y_pred)
    print(f"{name} - Confusion Matrix:")
    print(cm)
    
    logging.info(f"{name} - Classification Report:\n{pd.DataFrame(report).transpose()}")
    logging.info(f"{name} - Confusion Matrix:\n{cm}")

    return classifier


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
    plt.savefig(f"output/loss_curves_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.C}_{time.strftime('%Y-%m-%d %H-%M-%S')}.png", bbox_inches='tight')

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
    plt.savefig(f"output/probabilities_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.C}_{time.strftime('%Y-%m-%d %H-%M-%S')}.png", bbox_inches='tight')