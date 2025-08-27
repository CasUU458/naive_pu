from kiwisolver import strength
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from config import CONFIG
from numpy import random as rng

def reorder_dataframe_with_target_at_end(df):
    df_cols = df.columns.tolist()
    df_cols.remove('target')
    df_cols.append('target')

    return df[df_cols]


def set_positive_label_distribution(label_distribution, positives, negatives, random_state=CONFIG.SEED):
    # label_distribution = how balanced or imbalanced the labels should be.
    # this is used to replicate the distribution of the original experiment on another datasets.
    # always keep the size of test_negatives, and downsample the size of test_positives

    test_negatives_size = negatives.shape[0]

    n_positives_requested = int(test_negatives_size / (1 - label_distribution) * label_distribution)

    if n_positives_requested <= positives.shape[0]:
        return positives.sample(n=n_positives_requested, random_state=random_state)
    else:
        return positives


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


def SCAR(df, c,random_state=CONFIG.SEED):
    df["PU"] = df.loc[df['target'] == 1, 'target'].sample(frac=c, random_state=random_state)
    df["PU"] = df["PU"].fillna(0)
    df["PU"] = df["PU"].astype(int)

    return df.reset_index(drop=True)


def SAR(df, c,n_features=1,strength=10,random_state=CONFIG.SEED):
    # P(s =1 "| X")
    rng = np.random.default_rng(random_state)

    # if n_features > df.shape[1]-1:
    #     raise ValueError("Error: n_features is larger than the number of features in the DataFrame.")
    #     # print(f"Warning: n_features is larger than the number of features in the DataFrame. Setting n_features from {n_features} to {df.shape[1]-1}.")
    #     # n_features = df.shape[1]-1
    
    


    if n_features == 1:
        features = df.select_dtypes(include=[np.number]).drop(columns=["target"]).sample(n=1, axis=1, random_state=random_state).columns.tolist()
        CONFIG.dominant_features = features
        feature = features[0]
        values = df.loc[df["target"]>0,feature].astype(float).to_numpy()
        #boolean choice to go reverse importance
        # if np.random.rand() < 0.5:
        #     idx = idx[::-1]
        # values = (values - values.min()) / (values.max() - values.min())  # normalize to [0,1]
        n_values = len(values)
        modifier = np.ones(n_values)
        modifier *=strength
        modifier /= np.sqrt(len(values))
        probs = 1./(1+np.exp(-(np.transpose(values) * modifier)))
        probs = pd.Series(probs,index=df.loc[df["target"]>0].index)
        probs = probs.clip(lower=1e-20, upper=1-1e-20)
        
        
 
    
    else:
        

        n_clusters = strength #This may be bullshit
        n_features = n_features

        clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(df.drop(columns=["target"]))
        q = rng.uniform(0,1,size=(n_clusters,n_features)) #Probablity per cluster to occcur in feature X
        Xe = np.zeros((df.shape[0],n_features)) #Additional Xe features will be added to the dataset

        for g in range(n_clusters):
            mask = (clusters == g) 
            Xe[mask,:] = rng.binomial(1, q[g,:], size=(mask.sum(), n_features))

        feature_names = [f"Xe_{i}" for i in range(n_features)]
        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(Xe, columns=feature_names)], axis=1)
        CONFIG.dominant_features = feature_names
        

        p_low = 0.2
        p_high =0.8 # As accordint to paper
        
        output = 1
        for i in range(n_features):
            output *= (p_low**(1-Xe[:,i]) * p_high**(Xe[:,i])) 

        probs = output**(1/n_features)
        probs = probs[df.loc[df["target"]>0].index]
        probs = pd.Series(probs,index=df.loc[df["target"]>0].index)


    requested_n = int(len(probs) * c)
    # idx = probs.index.to_numpy()

    # if requested_n < len(probs):
    #     probs = probs / probs.sum()
    #     sampled = rng.choice(idx, size=requested_n, replace=False, p=probs)
    # else:
    #     raise ValueError("Error: requested label_distribution is larger than the number of positives.")
    
    df["PU"] = 0
    sampled_idx = df.loc[df["target"]>0].sample(n=requested_n,weights=probs, random_state=random_state).index
    df.loc[sampled_idx, "PU"] = 1

    # simple
    # feature = df.select_dtypes(include=[np.number]).drop(columns=["target"]).sample(n=1, axis=1, random_state=CONFIG.SEED).columns.tolist()
    # CONFIG.dominant_features = feature
    # df_pos = df.loc[df['target'] == 1]
    # df_pos = df_pos.sort_values(by=feature, ascending=False).iloc[-int(len(df_pos) * c):]
    # df_pos_ix = df_pos.index
    # df["PU"] = 0
    # df.loc[df_pos_ix, "PU"] = 1
    # df["PU"] = df["PU"].astype(int)

    return df.reset_index(drop=True)



def case_control(df, c,drop_feature=0,strength=10,random_state=CONFIG.SEED):
    
    rng = np.random.default_rng(random_state)

    feature = df.select_dtypes(include=[np.number]).drop(columns=["target"]).columns[drop_feature]
    values = df.loc[df["target"]>0,feature].astype(float).to_numpy()
    n_values = len(values)

    modifier = np.ones(n_values)
    modifier *=strength
    modifier /= np.sqrt(len(values))

    probs = 1./(1+np.exp(-(np.transpose(values) * modifier)))
    probs = pd.Series(probs,index=df.loc[df["target"]>0].index)
    requested_n = int(len(probs) * c)
    idx = probs.index.to_numpy()
    probs = probs / probs.sum()
    sampled_idx = rng.choice(idx, size=requested_n, replace=False, p=probs)
   
    df["PU"] = 0
    sampled_idx = df.loc[df["target"]>0].sample(n=requested_n,weights=probs, random_state=random_state).index
    df.loc[sampled_idx, "PU"] = 1
   
    df = df.drop(columns=feature)
    return df


