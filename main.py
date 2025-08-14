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
import logging

t = time.time()
print("Current time:", t)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',
                    filename= f"logs/{CONFIG.DATASET_NAME} {CONFIG.LABELING_MECHANISM} {CONFIG.c} {time.strftime('%Y-%m-%d %H-%M-%S')}.log",
                    filemode='w'
                    )


for C in CONFIG.to_dict().keys():
    logging.info(f"{C} {CONFIG.to_dict()[C]}")



data = get_pd_dataset(name='BreastCancer')
X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
                                                          test_size=CONFIG.TEST_SIZE,
                                                          c=CONFIG.c,
                                                          labeling_mechanism=CONFIG.LABELING_MECHANISM,
                                                          train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
                                                          test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
                                                          scale_data=CONFIG.SCALE_DATA)

print("Data loaded in {:.2f} seconds".format(time.time() - t))
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

logging.info(f"Test set shape: {X_test.shape, y_test.shape}")
logging.info(f"Train set shape: {X_train.shape, y_train.shape}")
logging.info(f"n labels train {y_train.sum()} n labels test {y_test.sum()}")

# Fit the Classic Logistic Regression model
clf = do_classification(ClassicLogReg(epochs=CONFIG.EPOCHS, learning_rate=CONFIG.LEARNING_RATE), "Classic Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Sklearn Logistic Regression model as a baseline
sk_clf = do_classification(SklearnLogisticRegression(penalty=None, max_iter=CONFIG.EPOCHS), "Sklearn Logistic Regression", X_train, y_train, X_test, y_test)

# Fit the Naive Logistic Regression model as a baseline
naive_clf = do_classification(NaiveLogReg(epochs=CONFIG.EPOCHS, learning_rate=CONFIG.LEARNING_RATE,learning_rate_c=CONFIG.LEARNING_RATE_C,c_estimate=CONFIG.INITIAL_GUESS_C), "Naive Logistic Regression", X_train, y_train, X_test, y_test)

plot_loss_curves(clf, naive_clf, c=CONFIG.c)
plot_probabilities(clf, naive_clf, X_test, y_test)
plt.show()
