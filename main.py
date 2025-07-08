import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from config import CONFIG
from data.metrics import plot_loss_curves, plot_probabilities, do_classification

matplotlib.use("TkAgg")
import time

from classifiers.classic_log_reg import ClassicLogReg
from classifiers.naive_log_reg import NaiveLogReg
from data.datasets import prepare_and_split_data, get_pd_dataset

t = time.time()
print("Current time:", t)

data = get_pd_dataset(name='BreastCancer')
X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
                                                          test_size=0.5,
                                                          c=CONFIG.c,
                                                          labeling_mechanism="SCAR",
                                                          train_label_distribution=None,
                                                          test_label_distribution=None,
                                                          scale_data="standard")

print("Data loaded in {:.2f} seconds".format(time.time() - t))
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

# Fit the Classic Logistic Regression model
clf = do_classification(ClassicLogReg(), "Classic Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Sklearn Logistic Regression model as a baseline
sk_clf = do_classification(SklearnLogisticRegression(penalty=None, max_iter=100000), "Sklearn Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Naive Logistic Regression model as a baseline
naive_clf = do_classification(NaiveLogReg(), "Naive Logistic Regression", X_train, y_train, X_test, y_test)

plot_loss_curves(clf, naive_clf, c=CONFIG.c)
plot_probabilities(clf, naive_clf, X_test, y_test)
plt.show()
