import os
import pandas as pd
import numpy as np
from data.datasets import get_pd_dataset,prepare_and_split_data
from data.dataset_helpers import SAR
from config import CONFIG, Config
import matplotlib.pyplot as plt
from classifiers.TM_log_reg import TwoModelLogReg
from classifiers.classic_log_reg import ClassicLogReg
from classifiers.naive_log_reg import NaiveLogReg
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import seaborn as sns
import numpy.random as rng
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
import json
import warnings
warnings.filterwarnings("ignore")  # suppress all warnings
from datetime import datetime
import sys
sys.path.append("/Users/cas/Documents/putm")
from putm import PUtm


global DATASETS #Datasets to use for experiments
global MODELS #Classifiers to use for experiments
global ITERS #Amount of iterations per experiment
global STEP_RESULT #Template for each iteration result
global RESULT_COLS #Columns for the result dataframe

global EXPERIMENT_VALUES #Values to use for the current experiment
global EXPERIMENT_ATTR  #Attribute to vary for the current experiment, must equal one of the CONFIG attributes

#2 var experiment_attr_2
global EXPERIMENT_ATTR_2
global EXPERIMENT_VALUES_2

global INCLUDE_ORACLE

os.makedirs("EXPERIMENTS", exist_ok=True)

def get_models():
    clf = {
        "classic": ClassicLogReg(),
        "naive": NaiveLogReg(),
        "oracle": ClassicLogReg()
    }

    clf_y = LogisticRegression(max_iter=CONFIG.max_iterations,penalty=CONFIG.penalty)
    clf_e = LogisticRegression(max_iter=CONFIG.max_iterations,penalty=CONFIG.penalty)
    TM = PUtm(clf=clf_y, clf_ex=clf_e,epochs=CONFIG.max_loop_iterations,epsilon=CONFIG.epsilon)
    clf["two_model"] = TM 

    return clf


def set_global_vars():
    global DATASETS
    DATASETS = {"mock":None,"diabetes":get_pd_dataset(name="diabetes"),"breastcancer":get_pd_dataset(name="breastcancer"),"mnist":None}

    global MODELS
    MODELS = get_models()

    global ITERS
    ITERS = np.arange(0,10,1)

    global STEP_RESULT
    STEP_RESULT = set_step_result()
    global RESULT_COLS
    RESULT_COLS = STEP_RESULT.index.tolist()

    global INCLUDE_ORACLE
    INCLUDE_ORACLE = True

def reset_config():
          # CONFIG.RANDOM_SEED = False
            CONFIG.device = 'cpu'
            CONFIG.dataset = 'mock' #MNIST, BreastCancer #mock or diabetes
            CONFIG.test_size = 0.2
            CONFIG.label_mechanism = 'SCAR_1_50'
            CONFIG.positive_ratio = None # number of positives / number of negatives or None to ignore
            CONFIG.scaler = "standard" # or "minmax"
            CONFIG.label_noise = None
            CONFIG.c = 0.2 # Labeling frequency


#Each iteration result template
def set_step_result():
    return pd.Series({
    "accuracy":0.0,
    "0 precision":0.0,
    "1 precision":0.0,
    "0 recall":0.0,
    "1 recall":0.0,
    "0 f1":0.0,
    "1 f1":0.0,
    "0 support":0,
    "1 support":0
}, name="results")

#Evaluate one iteration
def evaluate_step(y_test,y_pred):
    result_series = STEP_RESULT
    result_series["accuracy"] = accuracy_score(y_test, y_pred)
    result_series["0 precision"] = precision_score(y_test, y_pred, pos_label=0)
    result_series["1 precision"] = precision_score(y_test, y_pred, pos_label=1)
    result_series["0 recall"] = recall_score(y_test, y_pred, pos_label=0)
    result_series["1 recall"] = recall_score(y_test, y_pred, pos_label=1)
    result_series["0 f1"] = f1_score(y_test, y_pred, pos_label=0)
    result_series["1 f1"] = f1_score(y_test, y_pred, pos_label=1)
    result_series["0 support"] = sum(y_test == 0)
    result_series["1 support"] = sum(y_test == 1)
    return result_series

#Create the dataframe to store all experiment results, multiindex from all experiment parameters 
def make_test_result_df(sets,columns):    
    X = pd.MultiIndex.from_product(sets,names=columns).to_frame(index=False)
    X["accuracy"] = 0.0
    X["0 precision"] = 0.0
    X["1 precision"] = 0.0
    X["0 recall"] = 0.0
    X["1 recall"] = 0.0
    X["0 f1"] = 0.0
    X["1 f1"] = 0.0
    X["0 support"] = 0
    X["1 support"] = 0

    return X

# Run one experiment, varying the EXPERIMENT_ATTR over the EXPERIMENT_VALUES for all datasets, models and iterations
def experiment():
    path = "EXPERIMENTS"
    exp_path = os.path.join(path, f"{EXPERIMENT_ATTR}")
    os.makedirs(exp_path, exist_ok=True)
    result = make_test_result_df(sets=[EXPERIMENT_VALUES,DATASETS, ITERS, MODELS], columns=[EXPERIMENT_ATTR,"dataset","iter","model"])
    result.to_pickle(os.path.join(exp_path, "init.pkl"))

    result["calculated_c"] = 0.0
    result["feature"] = ""

    for exp in EXPERIMENT_VALUES:
        CONFIG.set_attr(EXPERIMENT_ATTR, exp)
        
        for i in ITERS:
            seed = int((i+1234)*1234)
            CONFIG.set_random_seed(seed)
            print(f"{i} {EXPERIMENT_ATTR}: {exp}, SEED: {seed}")
            
            for dataset_name,dataset in get_datasets(DATASETS).items():
                

                X_train,s_train,X_test,y_test = prepare_and_split_data(dataset)
                Y_TRAIN = CONFIG.true_train_labels

                for model_name,clf in MODELS.items():

                    # if model_name == "two_model" and dataset_name == "mnist":
                    #         continue #skip TM on mnist for now, takes too long

                    if not INCLUDE_ORACLE and model_name == "oracle":
                        continue #skip oracle model if do_oracle is False

                    try:
                        if model_name == "oracle":
                            clf.fit(X_train,Y_TRAIN) # Fit the oracle model using true labels
                        else:
                            clf.fit(X_train,s_train)
                        y_pred = clf.predict(X_test)
                        res = evaluate_step(y_test,y_pred)
                        mask =(
                            (result[EXPERIMENT_ATTR] == exp) &
                            (result["dataset"] == dataset_name) &
                            (result["iter"] == i) &
                            (result["model"] == model_name)
                        )
                        result.loc[mask,res.index] = res.values

                        if CONFIG.label_mechanism.lower().startswith("sar") or CONFIG.label_mechanism.lower().startswith("case"):
                            result.loc[mask,"calculated_c"] = CONFIG.calculated_c
                            result.loc[mask,"feature"] = f"{CONFIG.dominant_features}"

                    except Exception as e:
                        print(f"Error occurred for {model_name} on {dataset_name}: {e}")
        
        result.to_pickle(os.path.join(exp_path, f"TEMP_{exp}_{dataset_name}_{model_name}_{i}.pkl"))

    result.to_pickle(os.path.join(exp_path, f"final_{np.random.randint(1000)}.pkl"))
    return result

def double_experiment():
    path = "EXPERIMENTS"
    exp_path = os.path.join(path, f"{EXPERIMENT_ATTR}_{EXPERIMENT_ATTR_2}")
    os.makedirs(exp_path, exist_ok=True)
    result = make_test_result_df(sets=[EXPERIMENT_VALUES,EXPERIMENT_VALUES_2,DATASETS, ITERS, MODELS], columns=[EXPERIMENT_ATTR,EXPERIMENT_ATTR_2,"dataset","iter","model"])
    result.to_pickle(os.path.join(exp_path, "init.pkl"))
    result["calculated_c"] = 0.0
    result["feature"] = ""
    for exp in EXPERIMENT_VALUES:
        CONFIG.set_attr(EXPERIMENT_ATTR, exp)

        for exp2 in EXPERIMENT_VALUES_2:
            CONFIG.set_attr(EXPERIMENT_ATTR_2, exp2)

            for i in ITERS:
                seed = int((i+1234)*1234)
                CONFIG.set_random_seed(seed)
                print(f"{i} {EXPERIMENT_ATTR}: {exp}, SEED: {seed}")
                
                for dataset_name,dataset in get_datasets(DATASETS).items():
                
                    X_train,s_train,X_test,y_test = prepare_and_split_data(dataset)
                    Y_TRAIN = CONFIG.true_train_labels
                    for model_name,clf in MODELS.items():
                        if model_name == "two_model" and dataset_name == "mnist":
                            continue #skip TM on mnist for now, takes too long
                        try:
                            if model_name == "oracle":
                                clf.fit(X_train,Y_TRAIN)
                            else:
                                clf.fit(X_train,s_train)
                            
                            y_pred = clf.predict(X_test)
                            res = evaluate_step(y_test,y_pred)
                            mask =(
                                (result[EXPERIMENT_ATTR] == exp) &
                                (result["dataset"] == dataset_name) &
                                (result["iter"] == i) &
                                (result["model"] == model_name)
                            )
                            result.loc[mask,res.index] = res.values

                            if CONFIG.label_mechanism.lower().startswith("sar") or CONFIG.label_mechanism.lower().startswith("case"):
                                result.loc[mask,"calculated_c"] = CONFIG.calculated_c
                                result.loc[mask,"feature"] = f"{CONFIG.dominant_features}"

                        except Exception as e:
                            print(f"Error occurred for {model_name} on {dataset_name}: {e}")
        result.to_pickle(os.path.join(exp_path, f"TEMP_{exp}_{exp2}_{dataset_name}_{model_name}_{i}.pkl"))

    result.to_pickle(os.path.join(exp_path, f"final_{np.random.randint(1000)}.pkl"))
    return result

def get_mnist(seed):
    digits = str(np.random.default_rng(seed).choice(["1_7", "3_5", "3_8", "5_8", "6_9"], replace=False))
    name = f"mnist_{digits}"
    return get_pd_dataset(name=name)

def get_mock():
    return get_pd_dataset(name="mock")

def get_datasets(DATASETS):

    datasets = {}
    for key,item in DATASETS.items():
        if key == "mnist":
            datasets["mnist"] = get_mnist(CONFIG.random_state)
        elif key == "mock":
            datasets["mock"] = get_mock()
        else:
            datasets[key] = item
    return datasets

if __name__ == "__main__":

    # label_frequencies = [0.01,0.1,0.5,1.0]

    


    reset_config()
    set_global_vars()

    # DATASETS = {"mnist": None}
    # # clf_y = LogisticRegression(max_iter=CONFIG.max_iterations,penalty=CONFIG.penalty)
    # # clf_e = LogisticRegression(max_iter=CONFIG.max_iterations,penalty=CONFIG.penalty)
    # clf = ClassicLogReg()
    # MODELS = {"classic": clf}
    # label_frequencies = np.arange(0.1,1.1,0.1)
    # label_frequencies = np.concatenate(([0.01,0.05],label_frequencies))
    # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    # print("\n -- LABEL FREQUENCY -- \n")
    # try:
    #     INCLUDE_ORACLE = False
    #     EXPERIMENT_VALUES = label_frequencies
    #     EXPERIMENT_ATTR = "c"
    #     experiment()
    # except:
    #     print("Error occurred during LABEL FREQUENCY experiment")


    # DATASETS = {"mnist": None,"breastcancer":get_pd_dataset("breastcancer")}
    # clf_y = LogisticRegression()
    # clf_e = LogisticRegression()
    # clf = PUtm(clf=clf_y, clf_ex=clf_e,epochs=CONFIG.max_loop_iterations,epsilon=1e-5)
    # MODELS = {"two_model": clf}
    # label_frequencies = [0.005,0.01,0.02,0.04,0.05,0.1,0.2, 0.25,0.35,0.5,0.6,0.75,0.8, 1.0]
    # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    # print("\n -- LABEL FREQUENCY -- \n")
    # try:
    #     INCLUDE_ORACLE = False
    #     EXPERIMENT_VALUES = label_frequencies
    #     EXPERIMENT_ATTR = "c"
    #     experiment()
    # except:
    #     print("Error occurred during LABEL FREQUENCY experiment")
    
    # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))



    # reset_config()
    # set_global_vars()
    # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    # print("\n -- LABEL MECHANISM -- \n")
    # try:
    #     CONFIG.c  = 1
    #     tresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    #     exp_values= [f"SCAR_1_{t}" for t in tresholds]
    #     # EXPERIMENT_VALUES=["SCAR_1_1","SAR_1_0.0","SAR_1_0.2","SAR_1_0.4","SAR_1_0.6","SAR_1_0.8","SAR_1_0.9","SAR_1_0.95","SAR_4_5","SAR_5_4","casecontrol_0.5","casecontrol_0.75","casecontrol_0.9"]
    #     EXPERIMENT_VALUES = exp_values
    #     EXPERIMENT_ATTR = "label_mechanism"
    #     experiment()
    # except:
    #     print("Error occurred during label SCAR/SAR experiment")

    
    reset_config()
    # set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- LABEL DISTRIBUTION -- \n")

    try:
        CONFIG.c  = 0.1
        EXPERIMENT_VALUES= [0.1,0.2,0.3,0.4,0.5]
        EXPERIMENT_ATTR = "positive_ratio"
        experiment()
    except:
        print("Error occurred during label distribution experiment")

    # reset_config()
    # # set_global_vars()
    # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    # reset_config()
    # reset_config()
    # # print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    # print("\n -- SAR and LABEL DISTRIBUTION -- \n")
    # try:
    #     EXPERIMENT_VALUES=["SAR_1_0.0","SAR_1_0.2","SAR_1_0.4","SAR_1_0.6","SAR_1_0.8","SAR_1_0.9"]
    #     EXPERIMENT_ATTR = "label_mechanism"
    #     EXPERIMENT_VALUES_2= [0.1,0.2,0.3,0.4,0.5]
    #     EXPERIMENT_ATTR_2 = "c"
    #     double_experiment()
    # except:
    #     print("Error occurred during SAR and label distribution experiment")

    # postive ratio#


    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- label distribution and c -- \n")
    try:
        EXPERIMENT_VALUES= [0.05,0.1,0.2,0.3,0.4,0.5]
        EXPERIMENT_ATTR = "positive_ratio"
        EXPERIMENT_VALUES_2= [0.05,0.1,0.2,0.3,0.4,0.5]
        EXPERIMENT_ATTR_2 = "c"
        double_experiment()
    except:
        print("Error occurred during c and label distribution experiment")
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    reset_config()
    set_global_vars()

    # # LABEL NOISE #
    reset_config()
    set_global_vars()

    print("\n -- LABEL NOISE -- \n")
    try:
        CONFIG.c = 0.15
        EXPERIMENT_VALUES= [0.05,0.1,0.2,0.3,0.4,0.5]
        EXPERIMENT_ATTR = "label_noise"
        experiment()
    except:
        print("Error occurred during label noise experiment")

    # noise vs label frequency #
    reset_config()
    set_global_vars()

    print("\n -- NOISE VS LABEL FREQUENCY -- \n")
    try:
        EXPERIMENT_VALUES= [0.05,0.1,0.2,0.3,0.4,0.5]
        EXPERIMENT_ATTR = "label_noise"
        EXPERIMENT_VALUES_2= [0.05,0.1,0.2,0.4,0.6,0.8,1]
        EXPERIMENT_ATTR_2 = "c"
        double_experiment()
    except:
        print("Error occurred during noise vs label frequency experiment")