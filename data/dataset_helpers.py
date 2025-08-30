from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from config import CONFIG
import matplotlib.pyplot as plt

def reorder_dataframe_with_target_at_end(df):
    df_cols = df.columns.tolist()
    df_cols.remove('target')
    df_cols.append('target')

    return df[df_cols]


def set_positive_label_distribution(label_distribution, positives, negatives, random_state=None):
    # label_distribution = how balanced or imbalanced the labels should be.
    # this is used to replicate the distribution of the original experiment on another datasets.
    # always keep the size of test_negatives, and downsample the size of test_positives

    # set random_state if not provided
    if random_state is None:
        random_state = CONFIG.random_state

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


def SCAR(df, c,random_state=None):

    if random_state is None:
        random_state = CONFIG.random_state

    df["PU"] = df.loc[df['target'] == 1, 'target'].sample(frac=c, random_state=random_state)
    df["PU"] = df["PU"].fillna(0)
    df["PU"] = df["PU"].astype(int)

    return df.reset_index(drop=True)


def SAR(df, c,n_features=1,strength=10,random_state=None):
    
    if random_state is None:
        random_state = CONFIG.random_state

    # P(s =1 "| X")
    rng = np.random.default_rng(random_state)

    # if n_features > df.shape[1]-1:
    #     raise ValueError("Error: n_features is larger than the number of features in the DataFrame.")
    #     # print(f"Warning: n_features is larger than the number of features in the DataFrame. Setting n_features from {n_features} to {df.shape[1]-1}.")
    #     # n_features = df.shape[1]-1
    
    


    if n_features == 1:
        feature = str(rng.choice(df.drop(columns="target").select_dtypes(include=[np.number]).columns.to_list()))
        # features = df.select_dtypes(include=[np.number]).drop(columns=["target"]).sample(n=1, axis=1, random_state=random_state).columns.tolist()
        CONFIG.dominant_features = [feature]
        feature = feature
        values = df[feature].values
        n_samples = len(values)
        weight = np.ones(n_samples)
        weight *=strength
        weight /= np.sqrt(len(values))
        probs = 1./(1+np.exp(-(np.transpose(values) * weight))) #sigmoid, modifier
        # positive_indices = df.loc[df["target"]>0,"target"].index
        # probs = pd.Series(probs[positive_indices.values],index=positive_indices)
        # probs = probs.clip(lower=1e-7, upper=1-(1e-7))

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
        # probs = probs[df.loc[df["target"]>0].index]
        # probs = pd.Series(probs,index=df.loc[df["target"]>0].index)


    # requested_n = int(len(probs) * c)
    # idx = probs.index.to_numpy()

    # if requested_n < len(probs):
    #     probs = probs / probs.sum()
    #     sampled = rng.choice(idx, size=requested_n, replace=False, p=probs)
    # else:
    #     raise ValueError("Error: requested label_distribution is larger than the number of positives.")
    
    df["PU"] = 0
    idx = np.where(probs>0.8)[0] #Label all samples with prob > 0.8 as positive
    df = df.reset_index(drop=True)
    df.loc[idx, "PU"] = 1
    df.loc[df["target"]==0,"PU"] = 0 
    # df["PU"] = 0
    # df = df.reset_index(drop=True)
    # df.loc[df["PU"].sample(frac=c, weights=probs, random_state=random_state).index, "PU"] = 1 #Label positive samples according to probs
    # df.loc[df["target"]==0,"PU"] = 0
    CONFIG.SAR_c_log = np.round(df['PU'].sum()/df['target'].sum(),3)
    print(f"\n \n SAR: {CONFIG.SAR_c_log} of positive labels were selected, total = {df['PU'].sum()} \n \n")

    # simple
    # feature = df.select_dtypes(include=[np.number]).drop(columns=["target"]).sample(n=1, axis=1, random_state=CONFIG.SEED).columns.tolist()
    # CONFIG.dominant_features = feature
    # df_pos = df.loc[df['target'] == 1]
    # df_pos = df_pos.sort_values(by=feature, ascending=False).iloc[-int(len(df_pos) * c):]
    # df_pos_ix = df_pos.index
    # df["PU"] = 0
    # df.loc[df_pos_ix, "PU"] = 1
    # df["PU"] = df["PU"].astype(int)
    if n_features == 1:
        plt.figure()
        df_sorted = df[["PU",CONFIG.dominant_features[0]]].reset_index(drop=True)
        df_sorted["probs"] = probs
        df_sorted = df_sorted.sort_values(by=CONFIG.dominant_features[0], ascending=False)
        df_sorted["color"] = "C0"
        df_sorted.loc[df_sorted["PU"]==1,"color"] = "C1"
        plt.scatter(df_sorted.loc[df_sorted["PU"]==0,CONFIG.dominant_features[0]],df_sorted.loc[df_sorted["PU"]==0,"probs"], c="C0",label="Unlabeled", alpha=0.3,marker="x")

        plt.scatter(df_sorted.loc[df_sorted["PU"]==1,CONFIG.dominant_features[0]],df_sorted.loc[df_sorted["PU"]==1,"probs"], c="C1",label="Positive")
        plt.xlabel(f"{CONFIG.dominant_features[0]} feature value")
        plt.ylabel("Label probability")
        plt.title(f"SAR total positive labels: {df['PU'].sum()}, c: {CONFIG.SAR_c_log}")
        plt.legend()
        plt.savefig(f"SAR_feature_importance_{strength}.png")


    return df.reset_index(drop=True)



def case_control(df, c,drop_feature=0,strength=10,random_state=None):
    if random_state is None:
        random_state = CONFIG.random_state

    rng = np.random.default_rng(random_state)

    feature = df.select_dtypes(include=[np.number]).drop(columns=["target"]).columns[drop_feature]
    values = df[feature].values
    n_values = len(values)

    weight = np.ones(n_values)
    weight *= strength
    weight /= np.sqrt(len(values))

    probs = 1./(1+np.exp(-(np.transpose(values) * weight)))
    probs = pd.Series(probs)
   
    df["PU"] = 0
    idx = df.sample(frac=c, weights=probs, random_state=random_state).index
    df.loc[idx, "PU"] = 1
    df.loc[df["target"]==0,"PU"] = 0
    CONFIG.SAR_c_log = np.round(df['PU'].sum()/df['target'].sum(),3)
    #Make it case control
    df = df.drop(columns=feature)

    return df


