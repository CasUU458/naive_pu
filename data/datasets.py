import os
import tarfile
import pandas as pd
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sk, make_classification
from sklearn.datasets import fetch_openml
from data.dataset_helpers import (reorder_dataframe_with_target_at_end, set_positive_label_distribution,
                             normalize_data_standard_scalar, normalize_data_minmax_scalar, SCAR, SAR, case_control)

from config import CONFIG
import warnings
import tarfile
import numpy as np


def get_pd_dataset(name = CONFIG.DATASET_NAME):
    mnist_name = name.lower()
    name = mnist_name.split("_")[0]
    match name:
        case 'mock':
            return mock_dataset()
        case 'breastcancer':
            return load_breast_cancer()
        case 'mnist':
            return load_mnist(mnist_name)
        case 'diabetes':
            return load_diabetes()
        case _:
            raise NotImplementedError("Requested dataset not found")


def prepare_and_split_data(data,
                            test_size=None,
                            c=None,
                            labeling_mechanism=None,
                            label_distribution=None,
                            scale_data=None,
                            validation_frac=None,
                            random_seed=None,
                            as_numpy=True):
    
    #### load config
    test_size = test_size if test_size is not None else CONFIG.TEST_SIZE
    c = c if c is not None else CONFIG.c
    labeling_mechanism = labeling_mechanism if labeling_mechanism is not None else CONFIG.LABELING_MECHANISM
    label_distribution = label_distribution if label_distribution is not None else CONFIG.LABEL_DISTRIBUTION
    scale_data = scale_data if scale_data is not None else CONFIG.SCALE_DATA
    validation_frac = validation_frac if validation_frac is not None else CONFIG.VALIDATION_FRAC
    random_seed = random_seed if random_seed is not None else CONFIG.SEED

    test_size = float(test_size)

    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    assert 'target' in data.columns, "Data must contain a 'target' column"

    data = reorder_dataframe_with_target_at_end(data)

    CONFIG.true_prior_proba = data['target'].sum() / len(data)

    # ensure both labels are included in the test set by taking a fraction according to test_size of each label
    test_positives = data[data['target'] == 1].sample(frac=test_size, random_state=random_seed)
    test_negatives = data[data['target'] == 0].sample(frac=test_size, random_state=random_seed)

    # assert test_negatives.shape[0] + test_positives.shape[0] == int(test_size * len(data)), f"Test set size does not match expected size {test_negatives.shape[0] + test_positives.shape[0]} vs {int(test_size * len(data))}"

    if label_distribution is not None:
        test_positives = set_positive_label_distribution(label_distribution, test_positives, test_negatives, random_state=random_seed)

    test = pd.concat([test_positives, test_negatives]).sample(frac=1, random_state=random_seed)
    train = data.drop(test.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_positives = train[train['target'] == 1]
    train_negatives = train[train['target'] == 0]

    if label_distribution is not None:
        train_positives = set_positive_label_distribution(label_distribution, train_positives, train_negatives,random_state=random_seed)

    train = pd.concat([train_positives, train_negatives]).sample(frac=1, random_state=random_seed)

    match scale_data:
        case "standard":
            train, test = normalize_data_standard_scalar(train, test)
        case "minmax":
            train, test = normalize_data_minmax_scalar(train, test)
        case "none": # no scalar is used
            train, test = train, test
        case _:
            warnings.warn("Error: no scalar method specified")

    labeling_mechanism = labeling_mechanism.split("_")
    n_features = int(labeling_mechanism[1]) if len(labeling_mechanism) > 1 else 1
    strength = int(labeling_mechanism[2]) if len(labeling_mechanism) > 2 else 10
    labeling_mechanism = labeling_mechanism[0]
    match labeling_mechanism:

        case "SCAR":
            # c = p(s = 1 |y = 1) - thus, the probability that a positive label is labeled
            # following c, a fraction of the positive labels are unlabeled here (set to 0)
            train = SCAR(train, c,random_state=random_seed)
            test = SCAR(test, c,random_state=random_seed)
        case "SAR":
            train = SAR(train, c,n_features=n_features,strength=strength,random_state=random_seed)
            test = SAR(test, c,n_features=n_features,strength=strength,random_state=random_seed)
        case "casecontrol":
            drop_feature = np.random.default_rng(seed=random_seed).choice(train.shape[1]-1)
            train = case_control(train, c=c,drop_feature=drop_feature, strength=strength, random_state=random_seed)
            test = case_control(test,   c=c,drop_feature=drop_feature, strength=strength, random_state=random_seed)
        case _:
            raise ValueError("Error: specify correct scalar method")

    # train labels are the PU labels, test labels are the true labels
    if validation_frac is not None:
        validation = train.sample(frac=validation_frac, random_state=random_seed)
        train = train.drop(validation.index)
        X_validation = validation.drop(columns=['target', 'PU'])
        s_validation = validation['PU']
        y_validation = validation['target']
        VAL = [X_validation.values, y_validation.values, s_validation.values]

    X_train = train.drop(columns=['target', 'PU'])
    y_train = train['PU']
    X_test = test.drop(columns=['target', 'PU'])
    y_test = test['target']

    # store the original class priors for later reference
    CONFIG.train_prior_proba = train["target"].sum() / len(train)
    CONFIG.test_prior_proba = test["target"].sum() / len(test)

    # store the true train and PU test labels for later reference
    CONFIG.true_train_labels = train['target'].values
    CONFIG.PU_test_labels = test['PU'].values

    if as_numpy:
        X_train = X_train.values
        y_train = y_train.values
        X_test = X_test.values
        y_test = y_test.values

    if validation_frac is not None:
        return X_train, y_train, X_test, y_test, VAL
    
    return X_train, y_train, X_test, y_test


def mock_dataset(random_state=CONFIG.SEED):
    n_positives = 65
    n_negatives = int(n_positives*1/(1/(50)))
    data = make_classification(n_samples=2*n_negatives, n_features=4, n_informative=4, n_redundant=0, random_state=random_state)
    x = pd.DataFrame(data[0], columns=[f"feature_{i}" for i in range(4)])
    x['target'] = data[1]
    x_positive = x[x['target'] == 1]
    x_negative = x[x['target'] == 0]
    x_positive = x_positive.sample(n=n_positives, random_state=random_state)
    x = pd.concat([x_positive, x_negative]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return x
    # x = pd.DataFrame(x[0], columns=[f"feature_{i}" for i in range(20)])
    # x['target'] = x.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)


def load_breast_cancer():
    if not os.path.exists('breast_cancer.parquet'):
        data = load_breast_cancer_sk()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df.to_parquet('breast_cancer.parquet', engine='pyarrow')
    else:
        df = pd.read_parquet('breast_cancer.parquet', engine='pyarrow')
    return df


def load_mnist(name):
    if not os.path.exists('mnist_784.parquet'):
        # Fetch the MNIST dataset from OpenML and save it as a DataFrame
        mnist = fetch_openml('mnist_784', version=1, as_frame=True)
        # Save the DataFrame to a parquet file
        df = mnist.frame
        df.to_parquet('mnist_784.parquet', engine='pyarrow')
    else:
        # Load the MNIST dataset from the saved parquet file
        df = pd.read_parquet('mnist_784.parquet', engine='pyarrow')
    digits = None
    names = name.split("_")
    if len(names) == 3:
        digits = (names[1], names[2])
        

    df.rename(columns={'class': 'target'}, inplace=True)  # Rename the target column to 'target'
    df["target"] = df["target"].values.to_numpy()
    if digits is not None:
        d1 = digits[0]
        d2 = digits[1]
    else:
        d1 = "3"
        d2 = "5"

    df1 = df[df['target'] == d1]
    df2 = df[df['target'] == d2]

    # Concatenate the two digits and shuffle the DataFrame
    df = pd.concat([df1, df2], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    # Convert the target column to numeric values
    df.loc[df["target"] == d1, "target"] = 0
    df.loc[df["target"] == d2, "target"] = 1
    df["target"] = df["target"].astype(int)
    return df

def load_diabetes():
    file_name = "diabetes.csv"
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    df.rename(columns={'Outcome': 'target'}, inplace=True)
    return df

