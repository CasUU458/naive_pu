import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from config import CONFIG
from data.metrics import plot_loss_curves, plot_probabilities, do_classification,plot_metric_bar,plot_feature_weights,plot_validation
from data.datasets import load_diabetes
matplotlib.use("TkAgg")
import time

from classifiers.classic_log_reg import ClassicLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.TM_log_reg import TwoModelLogReg
from data.datasets import prepare_and_split_data, get_pd_dataset
import logging
import os

import sys
sys.path.append("/Users/cas/Documents/putm")
from putm import PUtm
from sklearn.linear_model import LogisticRegression
import json
import warnings
warnings.filterwarnings("ignore")  # suppress all warnings

def experiment():
    t = time.time()
    print("Current time:", t)

    #import config settings from json file
    CONFIG.set_random_seed(seed=42)  # set the random seed for reproducibility

    #check if logs directory exists, if not exist create it
    log_path = f"logs/{time.strftime('%Y-%m-%d %H-%M-%S')}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    """
    logs the configuration settings and results to a folder withing /logs
    """
    logging.basicConfig(level=logging.INFO,
                        format='%'
                        '(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S',
                        filename= f"{log_path}/{CONFIG.dataset} {CONFIG.label_mechanism} {CONFIG.c}.log",
                        filemode='w'
                        )

    #add the config parameters to the log
    for C in CONFIG.to_dict().keys():
        logging.info(f"{C} {CONFIG.to_dict()[C]}")


    #load the dataset
    data = get_pd_dataset(name=CONFIG.dataset)

    #preprocess and split the dataset into train an test data
    X_train, y_train, X_test, y_test,VAL = prepare_and_split_data(data = data,
                                                            test_size=CONFIG.test_size,
                                                            c=CONFIG.c,
                                                            label_mechanism=CONFIG.label_mechanism,
                                                            positive_ratio=CONFIG.positive_ratio,
                                                            scaler=CONFIG.scaler,validation_frac=0.2,as_numpy=False)

    print("Data loaded in {:.2f} seconds".format(time.time() - t))
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    print("Validation set shape:", VAL[0].shape, VAL[1].shape, VAL[2].shape)

    logging.info(f"Test set shape: {X_test.shape, y_test.shape}")
    logging.info(f"Train set shape: {X_train.shape, y_train.shape}")
    logging.info(f"n labels train {y_train.sum()} n labels test {y_test.sum()}")
    logging.info(f"Validation set shape: {VAL[0].shape, VAL[1].shape, VAL[2].shape}")

    # Fit the Classic Logistic Regression model
    clf = ClassicLogReg()
    clf = do_classification(clf, "Classic Logistic Regression", X_train, y_train, X_test, y_test)

    # Fit the Sklearn Logistic Regression model as a baseline

    TM_clf = TwoModelLogReg(validation=VAL)
    TM_clf = do_classification(TM_clf, "Two Model Logistic Regression", X_train, y_train, X_test, y_test)
    # sk_clf = do_classification(SklearnLogisticRegression(penalty=None, max_iter=CONFIG. max_iterations), "Sklearn Logistic Regression", X_train, y_train, X_test, y_test)
    # TM_clf = None
    # Fit the Naive Logistic Regression model as a baseline
    naive_clf = do_classification(NaiveLogReg(), "Naive Logistic Regression", X_train, y_train, X_test, y_test)
    return clf,naive_clf, TM_clf,X_test, y_test, log_path

def evaluate(clf, naive_clf, TM_clf, X_test, y_test, log_path):

    plot_loss_curves(clf, naive_clf, c=CONFIG.c,path=log_path)
  

    clfs = [clf, naive_clf, TM_clf]
    clf_names = ["Classic Logistic Regression", "Naive Logistic Regression", "Two model logic Regression"]
    plot_probabilities(clf, naive_clf, X_test, y_test,name_1=clf_names[0],name_2=clf_names[1],path=log_path)
    # plot_probabilities(clf, TM_clf, X_test, y_test,name_1=clf_names[0],name_2=clf_names[2],path=log_path)

    plot_metric_bar(clfs, X_test, y_test, clf_names=clf_names, path=log_path)


    plot_feature_weights(naive_clf, X_test.columns.to_numpy(),name=clf_names[1],path=log_path)
    plot_feature_weights(TM_clf._get_y_clf(), X_test.columns.to_numpy(),name=clf_names[2],path=log_path)
    plot_feature_weights(TM_clf._get_e_clf(), X_test.columns.to_numpy(),name=f"{clf_names[2]} e(x)",path=log_path)
    plot_validation(TM_clf,path=log_path)
    plt.show()
    return 0




        



if __name__ == "__main__":

    clf,naive_clf, TM_clf,X_test, y_test, log_path = experiment()
    evaluate(clf,naive_clf, TM_clf,X_test, y_test, log_path)

    # CONFIG.from_json("config.json")

    # data = get_pd_dataset(name=CONFIG.dataset)

    # #preprocess and split the dataset into train an test data
    # X_train, y_train, X_test, y_test,VAL = prepare_and_split_data(data = data,
    #                                                         test_size=CONFIG.test_size,
    #                                                         c=CONFIG.c,
    #                                                         labeling_mechanism="SAR_4",
    #                                                         train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
    #                                                         test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
    #                                                         scale_data=CONFIG.SCALE_DATA,validation_frac=CONFIG.VALIDATION_FRAC)





    # data = get_pd_dataset(name=CONFIG.dataset)

    # #preprocess and split the dataset into train an test data
    # X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
    #                                                         test_size=CONFIG.test_size,
    #                                                         c=CONFIG.c,
    #                                                         labeling_mechanism=CONFIG.LABELING_MECHANISM,
    #                                                         train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
    #                                                         test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
    #                                                         scale_data=CONFIG.SCALE_DATA)

    # print(X_train.shape, y_train.shape)


    # clf = TwoModelLogReg()
    # clf = do_classification(clf, "Two Model Logistic Regression", X_train, y_train, X_test, y_test)
 
  