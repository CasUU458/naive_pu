import os
import pandas as pd
from sklearn.datasets import load_breast_cancer as load_breast_cancer_sk, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_openml


class DataSets:

    def __init__(self, name='MNIST', digits=None):
        self.name = name
        self.data = None

        self.true_prior = None
        self.train_prior = None
        self.test_prior = None
        self.true_train_labels = None
        self.PU_test_labels = None

        if name == 'mock':
            self.data = mock_dataset()
        elif name == 'BreastCancer':
            self.data = load_breast_cancer()
        elif digits is not None and name == 'MNIST':
            self.data = load_mnist(digits=digits)
        else:
            self.data = load_mnist()

        assert isinstance(self.data, pd.DataFrame), "Data should be a pandas DataFrame"

    def get_X_y(self, test_size=0.2,
                c=1, labeling_mechanism="SCAR",
                train_balance=None,
                test_balance=None,
                scale_data=None,
                random_state=42):

        data = self.data.copy()
        # Ensure data is a pandas DataFrame
        assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"

        # Ensure 'target' column exists and is at the end of the DataFrame
        assert 'target' in data.columns, "Data must contain a 'target' column"
        data_cols = data.columns.tolist()
        data_cols.remove('target')
        data_cols.append('target')
        data = data[data_cols]  # Reorder columns to have 'target' at the end

        # determine class prior, for later reference
        self.true_prior = data['target'].sum() / len(data)

        # split the data into train and test set, test_size is the fraction of the postives that will be used for
        # testing negatives will be sampled in the same proportion as the positives, unless test_balance is specified
        # then the negatives will be sampled according to the test_balance ratio, i.e. if test_balance=0.2 then for
        # every positive in the test set there will be 5 negatives
        test_positives = data[data['target'] == 1].sample(frac=test_size, random_state=random_state)
        test_negatives = data[data['target'] == 0].sample(frac=test_size, random_state=random_state)

        if test_balance is not None:
            n_requested = int(test_negatives.shape[0] * test_balance)
            if n_requested < test_positives.shape[0]:
                test_positives = test_positives.sample(n=n_requested, random_state=random_state)
            else:
                print(
                    "Warning: requested test_balance is larger than the number of positives in the test set. Using all "
                    "positives.")

        # Concatenate the positive and negative samples to form the test set
        # and shuffle the rows to mix positives and negatives
        test = pd.concat([test_positives, test_negatives]).sample(frac=1, random_state=random_state)

        # drop the test instances from the original data
        # so that we can use the rest of the data for training
        train = data.drop(test.index)

        # reset the indexes
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # balance the training set, if train_balance is specified
        train_positives = train[train['target'] == 1]
        train_negatives = train[train['target'] == 0]
        if train_balance is not None:
            n_requested = int(train_negatives.shape[0] * train_balance)

            if n_requested < train_negatives.shape[0]:
                train_positives = train_positives.sample(n=n_requested, random_state=random_state)
            else:
                print("Warning: requested train_balance is larger than the number of positives in the training set."
                      " Using all negatives.")

        train = pd.concat([train_positives, train_negatives]).sample(frac=1, random_state=random_state)

        # scale the data
        if scale_data == "standard":
            scaler = StandardScaler()
            train[train.columns[:-1]] = scaler.fit_transform(train[train.columns[:-1]])
            test[test.columns[:-1]] = scaler.transform(test[test.columns[:-1]])

        elif scale_data == "minmax":
            scaler = MinMaxScaler()
            train[train.columns[:-1]] = scaler.fit_transform(train[train.columns[:-1]])
            test[test.columns[:-1]] = scaler.transform(test[test.columns[:-1]])

        if labeling_mechanism == "SCAR":
            # Sample c fraction of the positive class for PU learning completely randomly
            train["PU"] = train['target'].sample(frac=c, random_state=random_state)
            train["PU"] = train["PU"].fillna(0)
            train["PU"] = train["PU"].astype(int)
            test["PU"] = test['target'].sample(frac=c, random_state=random_state)
            test["PU"] = test["PU"].fillna(0)
            test["PU"] = test["PU"].astype(int)

        else:
            raise NotImplementedError("Please implement the labeling mechanism")

        # train labels are the PU labels, test labels are the true labels
        X_train = train.drop(columns=['target', 'PU'])
        y_train = train['PU']
        X_test = test.drop(columns=['target', 'PU'])
        y_test = test['target']

        # store the class priors for later reference    
        self.train_prior = train["target"].sum() / len(train)
        self.test_prior = test["target"].sum() / len(test)

        # store the true train and PU test labels for later reference
        self.true_train_labels = train['target'].values
        self.PU_test_labels = test['PU'].values

        return X_train, y_train, X_test, y_test


# P(s =1 "| X")
# to be implemented labeling mechanism
def SAR(df,
        feature_col,
        label_col="Target",
        prob_func=None,
        random_state=42) -> pd.DataFrame:
    raise NotImplementedError("Please implement the SAR algorithm for your specific dataset.")


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
        d1 = df[df['target'] == digits[0]]
        d2 = df[df['target'] == digits[1]]
    else:
        d1 = df[df['target'] == '3']
        d2 = df[df['target'] == '5']

    # Concatenate the two digits and shuffle the DataFrame
    df = pd.concat([d1, d2], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    # Convert the target column to numeric values
    df.loc[df["target"] == '3', "target"] = 0
    df.loc[df["target"] == '5', "target"] = 1
    return df
