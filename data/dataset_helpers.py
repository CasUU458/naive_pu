from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from config import CONFIG


def reorder_dataframe_with_target_at_end(df):
    df_cols = df.columns.tolist()
    df_cols.remove('target')
    df_cols.append('target')

    return df[df_cols]


def set_positive_label_distribution(label_distribution, positives, negatives):
    # label_distribution = how balanced or imbalanced the labels should be.
    # this is used to replicate the distribution of the original experiment on another datasets.
    # always keep the size of test_negatives, and downsample the size of test_positives

    test_negatives_size = negatives.shape[0]

    n_positives_requested = int(test_negatives_size / (1 - label_distribution) * label_distribution)

    if n_positives_requested <= positives.shape[0]:
        return positives.sample(n=n_positives_requested, random_state=CONFIG.SEED)
    else:
        return Exception(
            "Error: requested label_distribution is larger than the number of positives in the test set.")


def normalize_data_standard_scalar(train, test):
    scaler = StandardScaler()

    train[train.columns[:-1]] = scaler.fit_transform(train[train.columns[:-1]])
    test[test.columns[:-1]] = scaler.transform(test[test.columns[:-1]])

    return train, test


def normalize_data_minmax_scalar(train, test):
    scaler = MinMaxScaler()

    train[train.columns[:-1]] = scaler.fit_transform(train[train.columns[:-1]])
    test[test.columns[:-1]] = scaler.transform(test[test.columns[:-1]])

    return train, test


def SCAR(df, c):
    df["PU"] = df.loc[df['target'] == 1, 'target'].sample(frac=c, random_state=CONFIG.SEED)
    df["PU"] = df["PU"].fillna(0)
    df["PU"] = df["PU"].astype(int)

    return df


def SAR(df, c,n_features=1):
    # P(s =1 "| X")

    if n_features > df.shape[1]-1:
        raise ValueError("Error: n_features is larger than the number of features in the DataFrame.")
        # print(f"Warning: n_features is larger than the number of features in the DataFrame. Setting n_features from {n_features} to {df.shape[1]-1}.")
        # n_features = df.shape[1]-1
    features = df.select_dtypes(include=[np.number]).drop(columns=["target"]).sample(n=n_features, axis=1, random_state=CONFIG.SEED).columns.tolist()
    print(f"Using features: {features} for SAR labeling mechanism with c={c}")
    X = df[features]
    X = X.fillna(0)
    scores = X.abs().mean(axis=1)
    scores -= scores.min()
    scores /= scores.max()
    
    print(scores.mean())
    rng = np.random.default_rng(CONFIG.SEED)
    
    

    pos_idx = df[df['target'] == 1].index
    scores = scores.loc[pos_idx]
    requested_n = int(len(scores) * c)
    scores /= scores.sum()
    if requested_n < len(scores):
        sampled = rng.choice(scores.index, size=requested_n, replace=False, p=scores)
    else:
        sampled = scores.index
    
    df["PU"] = 0
    df.loc[sampled, "PU"] = 1
    # df.loc[(df["PU"] ==1) & (df["target"] == 0), "PU"] = 0 #only positive samples
    df["PU"] = df["PU"].astype(int)
    return df

