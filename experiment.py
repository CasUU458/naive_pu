import os
import pandas as pd
import numpy as np
from data.datasets import get_pd_dataset,prepare_and_split_data
from data.dataset_helpers import SAR
from config import CONFIG
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

os.makedirs("EXPERIMENTS", exist_ok=True)

def set_global_vars():
    global DATASETS
    DATASETS = ["mock","diabetes","breastcancer","mnist"]
    
    global MODELS
    MODELS = {
    "classic": ClassicLogReg(),
    "naive": NaiveLogReg(),
    "two_model": TwoModelLogReg()
    }

    global ITERS
    ITERS = np.arange(0,10,1)

    global STEP_RESULT
    STEP_RESULT = set_step_result()
    global RESULT_COLS
    RESULT_COLS = STEP_RESULT.index.tolist()

def reset_config():
    CONFIG.label_frequency = 0.2,
    CONFIG.DATASET_NAME = "mock",
    CONFIG.TEST_SIZE = 0.2,
    CONFIG.c = 0.2 #label frequency
    CONFIG.LABELING_MECHANISM = "SCAR",
    CONFIG.LABEL_DISTRIBUTION = None,
    CONFIG.SCALE_DATA  = "standard",
    CONFIG.EPOCHS = 300,
    CONFIG.INITIAL_GUESS_C = None,
    CONFIG.LEARNING_RATE = 0.001,
    CONFIG.LEARNING_RATE_C_modifier = 1,
    CONFIG.penalty = "l2",
    CONFIG.solver = "adam",
    CONFIG.VALIDATION_FRAC = None,
    CONFIG.TM_ALPHA = None
    CONFIG.SEED = 42
    CONFIG.CONVERGENCE_TOLERANCE = 1e-5

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
    
    for exp in EXPERIMENT_VALUES:
        CONFIG.set_attr(EXPERIMENT_ATTR, exp)
        
        for i in ITERS:
            seed = int((i+1234)*1234)
            CONFIG.set_random_seed(seed)
            print(f"{i} {EXPERIMENT_ATTR}: {exp}, SEED: {seed}")
            
            for dataset in DATASETS:
                data = get_pd_dataset(name=dataset)
                X_train,s_train,X_test,y_test = prepare_and_split_data(data)
        
                for name,clf in MODELS.items():
                    try:
                        clf.fit(X_train,s_train)
                        y_pred = clf.predict(X_test)
                        res = evaluate_step(y_test,y_pred)
                        mask =(
                            (result[EXPERIMENT_ATTR] == exp) &
                            (result["dataset"] == dataset) &
                            (result["iter"] == i) &
                            (result["model"] == name)
                        )
                        result.loc[mask,res.index] = res.values
                        result.to_pickle(os.path.join(exp_path, f"{exp}_{dataset}_{name}_{i}.pkl"))
                    except Exception as e:
                        print(f"Error occurred for {name} on {dataset}: {e}")
    result.to_pickle(os.path.join(exp_path, "final.pkl"))
    return result

def double_experiment():
    path = "EXPERIMENTS"
    exp_path = os.path.join(path, f"{EXPERIMENT_ATTR}_{EXPERIMENT_ATTR_2}")
    os.makedirs(exp_path, exist_ok=True)
    result = make_test_result_df(sets=[EXPERIMENT_VALUES,EXPERIMENT_VALUES_2,DATASETS, ITERS, MODELS], columns=[EXPERIMENT_ATTR,EXPERIMENT_ATTR_2,"dataset","iter","model"])
    result.to_pickle(os.path.join(exp_path, "init.pkl"))
    
    for exp in EXPERIMENT_VALUES:
        CONFIG.set_attr(EXPERIMENT_ATTR, exp)

        for exp2 in EXPERIMENT_VALUES_2:
            CONFIG.set_attr(EXPERIMENT_ATTR_2, exp2)

            for i in ITERS:
                seed = int((i+1234)*1234)
                CONFIG.set_random_seed(seed)
                print(f"{i} {EXPERIMENT_ATTR}: {exp}, SEED: {seed}")
                
                for dataset in DATASETS:
                    data = get_pd_dataset(name=dataset)
                    X_train,s_train,X_test,y_test = prepare_and_split_data(data)
            
                    for name,clf in MODELS.items():
                        try:
                            clf.fit(X_train,s_train)
                            y_pred = clf.predict(X_test)
                            res = evaluate_step(y_test,y_pred)
                            mask =(
                                (result[EXPERIMENT_ATTR] == exp) &
                                (result["dataset"] == dataset) &
                                (result["iter"] == i) &
                                (result["model"] == name)
                            )
                            result.loc[mask,res.index] = res.values
                            result.to_pickle(os.path.join(exp_path, f"{exp}_{exp2}_{dataset}_{name}_{i}.pkl"))
                        except Exception as e:
                            print(f"Error occurred for {name} on {dataset}: {e}")
    result.to_pickle(os.path.join(exp_path, "final.pkl"))
    return result




if __name__ == "__main__":

    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- LABEL FREQUENCY -- \n")
    try:
        EXPERIMENT_VALUES = np.arange(0.1,1.1,0.1)
        EXPERIMENT_ATTR = "c"
        experiment()
    except:
        print("Error occurred during LABEL FREQUENCY experiment")

    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- LABEL MECHANISM -- \n")
    try:
        CONFIG.c  = 0.2
        EXPERIMENT_VALUES=["SCAR_1_1","SAR_1_1","SAR_1_5","SAR_1_10","SAR_1_100","SAR_4_5","SAR_4_3","SAR_10_5"]
        EXPERIMENT_ATTR = "LABELING_MECHANISM"
        experiment()
    except:
        print("Error occurred during label SCAR/SAR experiment")

    
    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- LABEL DISTRIBUTION -- \n")

    try:
        CONFIG.c  = 0.2
        EXPERIMENT_VALUES= np.arange(0.1,0.55,0.05)
        EXPERIMENT_ATTR = "LABEL_DISTRIBUTION"
        experiment()
    except:
        print("Error occurred during label distribution experiment")

    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    
    print("\n -- LABEL MECHANISM CASECONTROL -- \n")
    try:
        CONFIG.c = 0.2
        EXPERIMENT_VALUES = ["casecontrol_1","casecontrol_5","casecontrol_10","case_control_100"]
        EXPERIMENT_ATTR = "LABELING_MECHANISM"
        experiment()
    except():
        print('Case control failed')

    reset_config()
    set_global_vars()
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))

    print("\n -- SAR and LABEL DISTRIBUTION -- \n")
    try:
        CONFIG.c  = 0.2
        EXPERIMENT_VALUES=["SCAR_1_1","SAR_1_1","SAR_1_5","SAR_1_10","SAR_1_100","SAR_4_5","SAR_4_3","SAR_10_5"]
        EXPERIMENT_ATTR = "LABELING_MECHANISM"
        EXPERIMENT_VALUES_2= np.arange(0.1,0.55,0.05)
        EXPERIMENT_ATTR_2 = "LABEL_DISTRIBUTION"
        double_experiment()
    except:
        print("Error occurred during SAR and label distribution experiment")
    print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z%z"))
