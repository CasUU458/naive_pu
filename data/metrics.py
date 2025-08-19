import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix,f1_score,accuracy_score, precision_score, recall_score
from config import CONFIG
import logging
import os

#fit the classifier and log its performance
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

    #logging
    logging.info(f"Training completed in {time.time() - t:.2f} seconds")
    logging.info(f"{name} - Classification Report:")
    logging.info(pd.DataFrame(report).transpose())
    logging.info(f"Confusion Matrix:")
    logging.info(cm)

    return classifier

# Plot loss curves for the classifiers
#Plots the log for the classic logistic regression and naive logistic regression
#For the naive it plots the loss for both the weight optimizer and c optimizer
#The true value and estimated value of c are also plotted
def plot_loss_curves(classic_log_reg, naive_log_reg, c=None, path="logs"):
    plt.figure(figsize=(12, 6))
    plt.plot(classic_log_reg.loss_log, label='Classic Log Reg Loss', color='blue', alpha=0.5)
    plt.plot(naive_log_reg.loss_log, label='Naive Log Reg Loss', color='orange', alpha=0.5)
    plt.plot(naive_log_reg.loss_c_log, label=r'Naive Log Reg $\hat{c}$ Loss', color='green', alpha=0.5)
    plt.plot(naive_log_reg.c_log, label=r"$\hat{c}$", color='red', alpha=0.8, linestyle='--')
    if c is not None:
        plt.axhline(y=c, color='purple', linestyle='--', label='True c')
    plt.title('Loss Curves')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path,f"loss_curves_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), bbox_inches='tight')

# Plot predicted probabilities for each of the instances in the test set
# Shows whether the model is correct in its predictions
# plot for both the classif clf and naive
def plot_probabilities(clf, naive_clf, X_test, y_test,name_1="Classic",name_2="Naive",path="logs"):
    x_range = np.arange(X_test.shape[0])

    plt.figure()
    # Classic Logistic Regression probabilities
    y_prob_classic = clf.predict_proba(X_test.values)

    sorted_indices = np.argsort(y_prob_classic)
    sorted_probs_classic = y_prob_classic[sorted_indices]
    sorted_y_test = y_test.values[sorted_indices]
    correct_pred = np.abs(sorted_probs_classic - sorted_y_test) <= 0.5
    correct_pred = np.where(correct_pred, 'C2', 'C3')
    plt.scatter(x_range, sorted_probs_classic, c=correct_pred, alpha=0.9,label=name_1, marker="*", s=15)
    plt.text(x_range[3], sorted_probs_classic[10]-0.04, f"Cl", fontsize=9, ha='left')


    # Naive Logistic Regression probabilities
    y_prob_naive = naive_clf.predict_proba(X_test.values)
    sorted_indices_naive = np.argsort(y_prob_naive)
    sorted_probs_naive = y_prob_naive[sorted_indices_naive]
    sorted_y_test_naive = y_test.values[sorted_indices_naive]
    correct_pred_naive = np.abs(sorted_probs_naive - sorted_y_test_naive) <= 0.5
    correct_pred_naive = np.where(correct_pred_naive, 'C0', 'C1')
    plt.scatter(x_range, sorted_probs_naive, c=correct_pred_naive,label=name_2,marker="o", alpha=0.9, s=15)
    plt.text(x_range[4], sorted_probs_naive[10]-0.04, f"PU", fontsize=9, ha='left')
    plt.title(f'Probabilities from Logistic Regression Models {name_1} and {name_2}')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Probability')
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.xticks([])
    plt.savefig(os.path.join(path,f"probabilities_{name_1}_{name_2}_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"), bbox_inches='tight')


def plot_metric_bar(classifiers, X_test, y_test, clf_names=None, path="logs"):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Default names if not provided
    if clf_names is None:
        clf_names = [f"clf_{i}" for i in range(len(classifiers))]

    # Collect metrics for each classifier
    metrics_values = []
    for clf in classifiers:
        y_pred = clf.predict(X_test.values)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
        metrics_values.append([accuracy, precision, recall, f1])

    metrics_values = np.array(metrics_values)  # shape: (n_classifiers, n_metrics)

    # Plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.15
    index = np.arange(len(metrics))

    for i, (clf_name, values) in enumerate(zip(clf_names, metrics_values)):
        bars = plt.bar(index + i * bar_width, values, bar_width, label=clf_name)

        # Add text labels on top of bars
        for bar, val in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha='center', va='bottom', fontsize=8
            )

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Classification Metrics')
    plt.xticks(index + bar_width * (len(classifiers) - 1) / 2, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            path,
            f"metrics_comparison_{'_'.join(clf_names)}_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"
        ),
        bbox_inches='tight'
    )


def plot_feature_weights(clf,feature_names,top_n=None,name="clf",path="logs"):
    bias = clf.get_bias()
    weights = clf.get_weights()
    weights_abs = np.abs(weights)
    assert len(weights) == len(feature_names), "Number of weights must match number of feature names"
    assert type(feature_names) == np.ndarray, "Feature names must be a numpy array"

    weights = np.insert(weights,0,bias)
    weights_abs= np.insert(weights_abs,0,np.abs(bias))
    feature_names = np.insert(feature_names, 0, "bias")

    df = pd.DataFrame({
        'feature': feature_names,
        'weight': weights,
        'abs_weight': weights_abs
    })

    df = df.sort_values(by='abs_weight', ascending=False)
    df = pd.concat([
        df[df["feature"] == "bias"],
        df[df["feature"] != "bias"]
    ])

    plt.figure(figsize=(10, 6))

    if top_n is None or top_n > len(df):
        top_n = len(df)

    plt.barh(df['feature'].iloc[:top_n], df['weight'].iloc[:top_n],color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance from Logistic Regression - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(path,f"feature_weights_{name}_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"), bbox_inches='tight')


def plot_validation(clf,path="logs"):
    if clf.val_log is None:
        logging.warning("No validation logs available.")

    val_log = clf.get_validation_logs()
    val_log = np.array(val_log)
    df = pd.DataFrame(val_log, columns=["name", "accuracy", "precision", "recall", "f1"])
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    y = df.loc[df["name"] == "y(X)"].reset_index(drop=True)
    e = df.loc[df["name"] == "e(X)"].reset_index(drop=True)
    threshold = df.loc[df["name"] == "threshold","accuracy"].reset_index(drop=True)
    or_ = df.loc[df["name"] == "OR","accuracy"].reset_index(drop=True)
    index = y.index
    # y(x)
    plt.figure(figsize=(10, 6))
    plt.plot(index,y["accuracy"], label="Accuracy", linestyle="--",alpha=0.3)
    plt.plot(index,y["precision"], label="Precision", linestyle="--",alpha=0.7)
    plt.plot(index,y["recall"], label="Recall", linestyle="--",alpha=0.7)
    plt.plot(index,y["f1"], label="F1-Score", linestyle="-",alpha=0.7,lw=2)
    plt.legend()
    plt.title(f"Validation Metrics for y(X) - {CONFIG.DATASET_NAME}")
    plt.tight_layout()
    plt.xlabel("epoch")
    plt.ylabel("Score")
    plt.ylim(0,1.1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(os.path.join(path,f"validation_metrics_y(X)_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"), bbox_inches='tight')

    # e(x)
    plt.figure(figsize=(10, 6))
    index = e.index
    plt.plot(index,e["accuracy"], label="Accuracy", linestyle="--",alpha=0.3)
    plt.plot(index,e["precision"], label="Precision", linestyle="--",alpha=0.7)
    plt.plot(index,e["recall"], label="Recall", linestyle="--",alpha=0.7)
    plt.plot(index,e["f1"], label="F1-Score", linestyle="-",alpha=0.7,lw=2)
    plt.legend()
    plt.title(f"Validation Metrics for e(X) - {CONFIG.DATASET_NAME}")
    plt.tight_layout()
    plt.xlabel("epoch")
    plt.ylabel("Score")
    plt.ylim(0,1.1)
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.savefig(os.path.join(path,f"validation_metrics_e(X)_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"), bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    index = threshold.index
    plt.plot(index,threshold, label="Threshold", linestyle="-",alpha=0.7)
    plt.legend()
    plt.title(f"TM Threshold - {CONFIG.DATASET_NAME}")
    plt.xlabel("epoch")
    plt.ylabel("Value")
    plt.savefig(os.path.join(path,f"TM_threshold_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"), bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    index = or_.index
    plt.plot(index, or_, label="OR", linestyle="-",alpha=0.7)
    plt.legend()
    plt.title(f"TM OR - {CONFIG.DATASET_NAME}")
    plt.xlabel("epoch")
    plt.ylabel("Value")
    plt.savefig(os.path.join(path,f"TM_OR_{CONFIG.DATASET_NAME}_{CONFIG.LABELING_MECHANISM}_{CONFIG.c}.png"), bbox_inches='tight')
