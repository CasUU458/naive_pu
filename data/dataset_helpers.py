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


def _1_feature_SAR(df,threshold,sig_weight,random_state,feature=None):

    rng = np.random.default_rng(random_state)
    if feature is None:
        feature = str(rng.choice(df.drop(columns=["target", "PU"]).select_dtypes(include=[np.number]).columns.to_list()))

    values = df[feature].values
    n_samples = len(values)
    weight = np.ones(n_samples)
    weight *= sig_weight
    weight /= np.sqrt(len(values))

    # shift = (values.max()-values.min())*threshold

    probs = 1./(1+np.exp(-(np.transpose(values) * weight))) #sigmoid, modifier
 
    CONFIG.PROBS.append(np.copy(probs)) #For development only
    # probs[probs<threshold] = 0

    df = df.reset_index(drop=True)
    # idx = rng.binomial(1, probs, size=df.shape[0]).astype(bool)
    idx = np.where(probs>=threshold)[0]    
    df.loc[idx, "PU"] = 1 #Select labels based on feature value independent of actual class label
    df.loc[df["target"]==0,"PU"] = 0 #Set negative samples back to 0, only positives may be labeled. 
    return df,feature


def _n_feature_SAR(df,n_features,threshold,n_clusters,random_state):

    rng = np.random.default_rng(random_state)
    n_clusters = n_clusters
    n_features = n_features

    clusters = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(df.drop(columns=["target"]))
    q = rng.uniform(0,1,size=(n_clusters,n_features)) #Probablity per cluster to occcur in feature X
    Xe = np.zeros((df.shape[0],n_features)) #Additional Xe features will be added to the dataset

    for g in range(n_clusters):
        mask = (clusters == g) 
        Xe[mask,:] = rng.binomial(1, q[g,:], size=(mask.sum(), n_features))

    feature_names = [f"Xe_{i}" for i in range(n_features)]
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(Xe, columns=feature_names)], axis=1)
    

    p_low = 0.2
    p_high =0.8 # As according to paper
    
    output = 1
    for i in range(n_features):
        output *= (p_low**(1-Xe[:,i]) * p_high**(Xe[:,i])) 

    probs = output**(1/n_features)

    df = df.reset_index(drop=True)
    probs[probs<threshold] = 0
    idx = rng.binomial(1, probs, size=df.shape[0]).astype(bool)
    df.loc[idx, "PU"] = 1
    df.loc[df["target"]==0,"PU"] = 0 
    return df,feature_names



def SAR(df,random_state,threshold,n_features=1,sig_weight=50,c=None,feature=None):


    if n_features == 1:
        df,feature = _1_feature_SAR(df,
                                    threshold=threshold,
                                    sig_weight=sig_weight,
                                    random_state=random_state,
                                    feature=feature
                                    )
    else:
        df,feature = _n_feature_SAR(df,
                                    n_features=n_features,
                                    threshold=0,
                                    n_clusters=int(threshold), #hack
                                    random_state=random_state)

    rng = np.random.default_rng(random_state)


    if c is not None:
        calculated_c = df['PU'].sum()/df['target'].sum()
        if calculated_c > c:
            n_requested = int(df["target"].sum() * c)
            n_to_remove = df['PU'].sum() - n_requested
            idx = df.loc[df["PU"] > 0].sample(n=n_to_remove,random_state=random_state).index
            df.loc[idx, 'PU'] = 0
            print(f"SAR: Removed {n_to_remove} positive labels to meet the upper bound of {c}, new sum {df['PU'].sum()}")

   


    return df.reset_index(drop=True),feature

def label_2_PU(df, mechanism,c,random_state=None,test=False):
    """ Converts the fully supervised labels to PU labels based on the specified label mechanism
    SCAR = select labels completely at random, c = fraction of postives to be selected
    SAR = select labels based on the feature importance, c cannot be specified but is a outcome of the selection process.
    casecontrol = same as SAR, but drop the feature from the dataset after labeling has been done. 
    
    for SAR and casecontrol the parameter c functions as an upperbound for the label frequency.
    If after SAR and control labeling the dataset the fraction of positives is still above c, we randomly select positives until we reach the upper bound.

    """
    df["PU"] = 0
    df = df.reset_index(drop=True)

    if random_state is None:
        random_state = CONFIG.random_state

    mechanism = mechanism.lower()
    mechanism_parts = mechanism.split("_")
    mechanism = mechanism_parts[0]

    match mechanism:
        case "scar":
            df = SCAR(df,c)
            feature = "--scar--"
        case "sar":
            n_features = int(mechanism_parts[1]) if len(mechanism_parts) > 1 else 1
            threshold = float(mechanism_parts[2]) 
            sigweight = float(mechanism_parts[3]) if len(mechanism_parts) > 3 else 100
            df,feature = SAR(df,
                               random_state=random_state,
                               n_features=n_features,
                               threshold=threshold,
                               sig_weight=sigweight,
                               c=c)
        case "casecontrol":
            threshold = float(mechanism_parts[1])
            feature = mechanism_parts[2] if len(mechanism_parts) > 2 else None
            if feature is None:
                feature = df.drop(columns=["target","PU"]).select_dtypes(include=[np.number]).columns.to_list()
                feature = str(np.random.default_rng(seed=random_state).choice(feature))
            df,feature = SAR(df,n_features=1,feature=feature,threshold=threshold,random_state=random_state,c=c)
            #Make it case control
            df.drop(columns=[feature],inplace=True)
        case _:
            raise ValueError(f"Error: label mechanism {mechanism} not recognized, use SCAR, SAR or casecontrol")
    
    calculated_c = df['PU'].sum()/df['target'].sum()
    if calculated_c < 0.01:
        print(f"\n SAR: Warning: Positive label fraction {calculated_c:.2f} is below 1% \n")
   
    return df,feature

def add_label_noise(train, noise, random_state=None):

    print("\n WARNING -- Adding label noise -- WARNING \n")
    if random_state is None:
        random_state = CONFIG.random_state

    noise_idx = train.sample(frac=noise, random_state=random_state).index
    train.loc[noise_idx, 'target'] = 1 - train.loc[noise_idx, 'target']
    
    return train
    



