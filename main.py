import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

import config 
from config import CONFIG
from data.metrics import plot_loss_curves, plot_probabilities, do_classification

matplotlib.use("TkAgg")
import time

from classifiers.classic_log_reg import ClassicLogReg
from classifiers.naive_log_reg import NaiveLogReg
from data.datasets import prepare_and_split_data, get_pd_dataset

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',
                    filename= f"{CONFIG.DATASET_NAME} {CONFIG.LABELING_MECHANISM} {CONFIG.C} {time.strftime('%Y-%m-%d %H-%M-%S')}.log",
                    filemode='w'
                    )


for C in CONFIG.to_dict().keys():
    logging.info(f"{C} {CONFIG.to_dict()[C]}")




t = time.time()
logging.info("Current time: %s", t)

#Available sets, BreastCancer, MNIST or mock
data = get_pd_dataset(name=CONFIG.DATASET_NAME)
X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
                                                          test_size=CONFIG.TEST_SIZE, # 20% for testing
                                                          c=CONFIG.C, #fraction of data to use
                                                          labeling_mechanism=CONFIG.LABELING_MECHANISM,
                                                          train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
                                                          test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
                                                          scale_data=CONFIG.SCALE_DATA)

print("Data loaded in {:.2f} seconds".format(time.time() - t))
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

logging.info("Train set shape: %s %s", X_train.shape, y_train.shape)
logging.info("Test set shape: %s %s", X_test.shape, y_test.shape)

logging.info("Train n positive: %s", y_train.value_counts().get(1, 0))
logging.info("Train n negative: %s", y_train.value_counts().get(0, 0))


# Fit the Classic Logistic Regression model
clf = do_classification(ClassicLogReg(penalty=CONFIG.PENALTY, epochs=CONFIG.EPOCHS,solver=CONFIG.SOLVER), "Classic Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Sklearn Logistic Regression model as a baseline
sk_clf = do_classification(SklearnLogisticRegression(penalty=CONFIG.PENALTY, max_iter=CONFIG.EPOCHS), "Sklearn Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Naive Logistic Regression model as a baseline
naive_clf = do_classification(NaiveLogReg(penalty=CONFIG.PENALTY,epochs=CONFIG.EPOCHS,solver=CONFIG.SOLVER,c_estimate=CONFIG.INITIAL_GUESS_C), "Naive Logistic Regression", X_train, y_train, X_test, y_test)

plot_loss_curves(clf, naive_clf, c=CONFIG.C)
plot_probabilities(clf, naive_clf, X_test, y_test)
plt.show()
