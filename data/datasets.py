import os
import pandas as pd
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sk, make_classification
from sklearn.datasets import fetch_openml
from data.dataset_helpers import (reorder_dataframe_with_target_at_end, set_positive_label_distribution,
                             normalize_data_standard_scalar, normalize_data_minmax_scalar, SCAR, SAR)

from config import CONFIG


def get_pd_dataset(name = CONFIG.DATASET_NAME, digits=None):
    match name:
        case 'mock':
            return mock_dataset()
        case 'BreastCancer':
            return load_breast_cancer()
        case 'MNIST':
            if digits is not None:
                return load_mnist(digits=digits)
            else:
                return load_mnist()
        case _:
            raise NotImplementedError("Requested dataset not found")


def prepare_and_split_data(data,
                           test_size=0.2,
                           c=1,
                           labeling_mechanism="SCAR",
                           train_label_distribution=None,
                           test_label_distribution=None,
                           scale_data=None,validation_frac=None):

    assert isinstance(data, pd.DataFrame), "Data must be a pandas DataFrame"
    assert 'target' in data.columns, "Data must contain a 'target' column"

    data = reorder_dataframe_with_target_at_end(data)

    CONFIG.true_prior_proba = data['target'].sum() / len(data)

    # ensure both labels are included in the test set by taking a fraction according to test_size of each label
    test_positives = data[data['target'] == 1].sample(frac=test_size, random_state=CONFIG.SEED)
    test_negatives = data[data['target'] == 0].sample(frac=test_size, random_state=CONFIG.SEED)

    assert test_negatives.shape[0] + test_positives.shape[0] == int(test_size * len(data)), f"Test set size does not match expected size {test_negatives.shape[0] + test_positives.shape[0]} vs {int(test_size * len(data))}"

    if test_label_distribution is not None:
        test_positives = set_positive_label_distribution(test_label_distribution, test_positives, test_negatives)

    test = pd.concat([test_positives, test_negatives]).sample(frac=1, random_state=CONFIG.SEED)
    train = data.drop(test.index)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train_positives = train[train['target'] == 1]
    train_negatives = train[train['target'] == 0]

    if train_label_distribution is not None:
        train_positives = set_positive_label_distribution(test_label_distribution, train_positives, train_negatives)

    train = pd.concat([train_positives, train_negatives]).sample(frac=1, random_state=CONFIG.SEED)

    match scale_data:
        case "standard":
            train, test = normalize_data_standard_scalar(train, test)
        case "minmax":
            train, test = normalize_data_minmax_scalar(train, test)
        case "none": # no scalar is used
            train, test = train, test
        case _:
            raise ValueError("Error: specify correct scalar method")

    labeling_mechanism = labeling_mechanism.split("_")
    n_features = int(labeling_mechanism[1]) if len(labeling_mechanism) > 1 else 1
    labeling_mechanism = labeling_mechanism[0]
    match labeling_mechanism:

        case "SCAR":
            # c = p(s = 1 |y = 1) - thus, the probability that a positive label is labeled
            # following c, a fraction of the positive labels are unlabeled here (set to 0)
            train = SCAR(train, c)
            test = SCAR(test, c)
        case "SAR":
            train = SAR(train, c,n_features=n_features)
            test = SAR(test, c,n_features=n_features)
        case _:
            raise ValueError("Error: specify correct scalar method")

    # train labels are the PU labels, test labels are the true labels
    if validation_frac is not None:
        validation = train.sample(frac=validation_frac, random_state=CONFIG.SEED)
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


    if validation_frac is not None:
        return X_train, y_train, X_test, y_test, VAL
    
    return X_train, y_train, X_test, y_test


def mock_dataset():
    data = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    x = pd.DataFrame(data[0], columns=[f"feature_{i}" for i in range(20)])
    x['target'] = data[1]
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


def load_mnist(digits: str | None = None):
    if not os.path.exists('mnist_784.parquet'):
        # Fetch the MNIST dataset from OpenML and save it as a DataFrame
        mnist = fetch_openml('mnist_784', version=1, as_frame=True)
        # Save the DataFrame to a parquet file
        df = mnist.frame
        df.to_parquet('mnist_784.parquet', engine='pyarrow')
    else:
        # Load the MNIST dataset from the saved parquet file
        df = pd.read_parquet('mnist_784.parquet', engine='pyarrow')
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

