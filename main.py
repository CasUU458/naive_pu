import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from config import CONFIG
from data.metrics import plot_loss_curves, plot_probabilities, do_classification,plot_metric_bar,plot_feature_weights,plot_validation

matplotlib.use("TkAgg")
import time

from classifiers.classic_log_reg import ClassicLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.TM_log_reg import TwoModelLogReg
from data.datasets import prepare_and_split_data, get_pd_dataset
import logging
import os

def experiment():
    t = time.time()
    print("Current time:", t)

    #import config settings from json file
    CONFIG.from_json("config.json")

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
                        filename= f"{log_path}/{CONFIG.DATASET_NAME} {CONFIG.LABELING_MECHANISM} {CONFIG.c}.log",
                        filemode='w'
                        )

    #add the config parameters to the log
    for C in CONFIG.to_dict().keys():
        logging.info(f"{C} {CONFIG.to_dict()[C]}")


    #load the dataset
    data = get_pd_dataset(name=CONFIG.DATASET_NAME)

    #preprocess and split the dataset into train an test data
    X_train, y_train, X_test, y_test,VAL = prepare_and_split_data(data = data,
                                                            test_size=CONFIG.TEST_SIZE,
                                                            c=CONFIG.c,
                                                            labeling_mechanism=CONFIG.LABELING_MECHANISM,
                                                            train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
                                                            test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
                                                            scale_data=CONFIG.SCALE_DATA,validation_frac=CONFIG.VALIDATION_FRAC)

    print("Data loaded in {:.2f} seconds".format(time.time() - t))
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    print("Validation set shape:", VAL[0].shape, VAL[1].shape, VAL[2].shape)

    logging.info(f"Test set shape: {X_test.shape, y_test.shape}")
    logging.info(f"Train set shape: {X_train.shape, y_train.shape}")
    logging.info(f"n labels train {y_train.sum()} n labels test {y_test.sum()}")
    logging.info(f"Validation set shape: {VAL[0].shape, VAL[1].shape, VAL[2].shape}")

    # Fit the Classic Logistic Regression model
    clf = do_classification(ClassicLogReg(epochs=CONFIG.EPOCHS, learning_rate=CONFIG.LEARNING_RATE,solver=CONFIG.solver,penalty=CONFIG.penalty), "Classic Logistic Regression", X_train, y_train, X_test, y_test)

    # Fit the Sklearn Logistic Regression model as a baseline

    TM_clf = TwoModelLogReg(epochs=CONFIG.EPOCHS, learning_rate=CONFIG.LEARNING_RATE,penalty=CONFIG.penalty,solver=CONFIG.solver,validation=VAL,alpha=CONFIG.TM_ALPHA)
    TM_clf = do_classification(TM_clf, "Two Model Logistic Regression", X_train, y_train, X_test, y_test)
    # sk_clf = do_classification(SklearnLogisticRegression(penalty=None, max_iter=CONFIG.EPOCHS), "Sklearn Logistic Regression", X_train, y_train, X_test, y_test)

    # Fit the Naive Logistic Regression model as a baseline
    naive_clf = do_classification(NaiveLogReg(epochs=CONFIG.EPOCHS, learning_rate=CONFIG.LEARNING_RATE,learning_rate_c=CONFIG.LEARNING_RATE_C,c_estimate=CONFIG.INITIAL_GUESS_C,solver=CONFIG.solver,penalty=CONFIG.penalty), "Naive Logistic Regression", X_train, y_train, X_test, y_test)
    return clf,naive_clf, TM_clf,X_test, y_test, log_path

def evaluate(clf, naive_clf, TM_clf, X_test, y_test, log_path):

    plot_loss_curves(clf, naive_clf, c=CONFIG.c,path=log_path)
  

    clfs = [clf, naive_clf, TM_clf]
    clf_names = ["Classic Logistic Regression", "Naive Logistic Regression", "Two model logic Regression"]
    plot_probabilities(clf, naive_clf, X_test, y_test,name_1=clf_names[0],name_2=clf_names[1],path=log_path)
    plot_probabilities(clf, TM_clf, X_test, y_test,name_1=clf_names[0],name_2=clf_names[2],path=log_path)

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

    # data = get_pd_dataset(name=CONFIG.DATASET_NAME)

    # #preprocess and split the dataset into train an test data
    # X_train, y_train, X_test, y_test,VAL = prepare_and_split_data(data = data,
    #                                                         test_size=CONFIG.TEST_SIZE,
    #                                                         c=CONFIG.c,
    #                                                         labeling_mechanism="SAR_4",
    #                                                         train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
    #                                                         test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
    #                                                         scale_data=CONFIG.SCALE_DATA,validation_frac=CONFIG.VALIDATION_FRAC)





    # data = get_pd_dataset(name=CONFIG.DATASET_NAME)

    # #preprocess and split the dataset into train an test data
    # X_train, y_train, X_test, y_test = prepare_and_split_data(data = data,
    #                                                         test_size=CONFIG.TEST_SIZE,
    #                                                         c=CONFIG.c,
    #                                                         labeling_mechanism=CONFIG.LABELING_MECHANISM,
    #                                                         train_label_distribution=CONFIG.TRAIN_LABEL_DISTRIBUTION,
    #                                                         test_label_distribution=CONFIG.TEST_LABEL_DISTRIBUTION,
    #                                                         scale_data=CONFIG.SCALE_DATA)

    # print(X_train.shape, y_train.shape)


    # clf = TwoModelLogReg()
    # clf = do_classification(clf, "Two Model Logistic Regression", X_train, y_train, X_test, y_test)
