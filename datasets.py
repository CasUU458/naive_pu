import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler





class DataSets():
    
    def __init__(self,name='BreastCancer'):
        self.name = name
        self.data = None

        self.true_prior = None
        self.train_prior = None
        self.test_prior = None
        self.true_train_labels = None
        self.PU_test_labels = None



        if name == 'DATASET_NAME':
            raise NotImplementedError("Please implement the dataset loading for your specific dataset.")
        
        # load breast cancer dataset if name is not recognized
        self.data = self._load_breast_cancer()
        assert isinstance(self.data, pd.DataFrame), "Data should be a pandas DataFrame"        
       


    
    def _load_breast_cancer(self):
        if not os.path.exists('breast_cancer.parquet'):
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df.to_parquet('breast_cancer.parquet',engine='pyarrow')             
        else:
            df = pd.read_parquet('breast_cancer.parquet', engine='pyarrow')
        return df

    def get_X_y(self, test_size=0.2,
    c=1,labeling_mechanism="SCAR",
    train_balance=None,
    test_balance=None,
    scale_data=None,
    random_state=42):
        
        data = self.data
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

        #split the data into train and test set, test_size is the fraction of the postives that will be used for testing
        #negatives will be sampled in the same proportion as the positives, unless test_balance is specified
        #then the negatives will be sampled according to the test_balance ratio, i.e. if test_balance=0.2 then for every positive in the test set there will be 5 negatives 
        test_positives  = data[data['target'] == 1].sample(frac=test_size, random_state=random_state) 
        if test_balance is not None:
            test_negatives = data[data['target'] == 0].sample(n=int(len(test_positives)*1/test_balance),random_state=random_state)
        else:
            test_negatives = data[data['target'] == 0].sample(frac=test_size, random_state=random_state)
        

        # Concatenate the positive and negative samples to form the test set
        # and shuffle the rows to mix positives and negatives
        test = pd.concat([test_positives, test_negatives]).sample(frac=1, random_state=random_state)
        
        # drop the test instances from the original data
        # so that we can use the rest of the data for training
        train = data.drop(test.index)

        #reset the indexes
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # balance the training set, if train_balance is specified
        train_positives = train[train['target'] == 1]
        if train_balance is not None:
            train_negatives = train[train['target'] == 0].sample(n=int(len(train_positives)*1/train_balance),random_state=random_state)
        else:
            train_negatives = train[train['target'] == 0].sample(frac=1, random_state=random_state)
        
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

#P(s =1 "| X")
# to be implemented labeling mechanism
def SAR(df,
    feature_col,
    label_col="Target",
    prob_func=None,
    random_state=42)-> pd.DataFrame:
    
    raise NotImplementedError("Please implement the SAR algorithm for your specific dataset.")