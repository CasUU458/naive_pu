import pandas as pd 
import os 
import numpy as np
from sklearn.datasets import make_classification,fetch_openml

PATH = "/Users/cas/Documents/naive_pu"

def get_n_labels(dataset,c_values):

    df = get_pd_dataset(dataset)
    class_prior = df["target"].mean()
    df = df.sample(frac=0.2)

    lib = {}
    for c in c_values:
        value = df.loc[df["target"]>0,"target"].sample(frac=c).sum()
        lib[c] = f"{c} | {int(value)}"
    lib["title"] = f"labels: {df.shape[0]}, class prior = {np.round(class_prior,2)}"

    return lib



def get_pd_dataset(name = None):
    mnist_name = name.lower()
    name = mnist_name.split("_")[0]
    match name:
        case 'mock':
            return mock_dataset()
        case 'breastcancer':
            return pd.read_parquet(os.path.join(PATH, 'breast_cancer.parquet'), engine='pyarrow')
        case 'mnist':
            return load_mnist(mnist_name)
        case 'diabetes':
            return load_diabetes()
        case _:
            raise NotImplementedError("Requested dataset not found")


def mock_dataset():
    random_state = 42 
    n_positives = 650
    n_negatives = int(n_positives*1.50)
    data = make_classification(n_samples=2*n_negatives, n_features=4, n_informative=4, n_redundant=0,flip_y=0.01,class_sep=1,random_state=random_state)
    x = pd.DataFrame(data[0], columns=[f"feature_{i}" for i in range(4)])
    x['target'] = data[1]
    x_positive = x[x['target'] == 1]
    x_negative = x[x['target'] == 0]
    x_positive = x_positive.sample(n=n_positives, random_state=random_state)
    x = pd.concat([x_positive, x_negative]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return x
    # x = pd.DataFrame(x[0], columns=[f"feature_{i}" for i in range(20)])
    # x['target'] = x.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)



def load_mnist(name):

    df = pd.read_parquet(os.path.join(PATH, 'mnist_784.parquet'), engine='pyarrow')
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
    file_name = os.path.join(PATH, 'diabetes.csv')
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None
    df.rename(columns={'Outcome': 'target'}, inplace=True)
    return df








