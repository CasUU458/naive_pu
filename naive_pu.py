import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification,load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
import os
import pyarrow  as pa

import torch
import time


class LogisticRegression():
    def __init__(self, learning_rate=0.005, num_iterations=1000,device='cpu',tolerance=1e-6):
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.device = device
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.tolerance = tolerance

    @staticmethod
    def _sigmoid(z):
        positive_mask = z >= 0
        negative_mask = ~positive_mask

        out = torch.zeros_like(z, dtype=torch.float32)
        out[positive_mask] = 1. / (1. + torch.exp(-z[positive_mask]))
        out[negative_mask] = torch.exp(z[negative_mask]) / (1. + torch.exp(z[negative_mask]))

        return out


    #loss function binary cross entropy or log likelihood?
    def _loss (self, y_true, y_pred):

        eps = 1e-15  # to avoid log(0), numerical stability, small constant
        # ensure y_pred is in the range [eps, 1-eps]
        #clamp_min y_pred to avoid log(0)

        #positive contribution
        y_pred = y_pred.clamp(eps, 1. - eps)
        term1 = y_true* torch.log(y_pred)
        term2 = (1. - y_true) * torch.log(1. - y_pred)
        loss = -torch.mean(term1 + term2)
        return loss
    

    def fit(self, X, y):
        num_samples, n_features = X.shape
        # self.weights = torch.zeros((num_features,1), dtype=torch.float32,requires_grad=True)
        # self.bias = torch.zeros(1, dtype=torch.float32,requires_grad=True)

        # y_t = torch.as_tensor(y, dtype=torch.float32)

        self.weights = torch.zeros(n_features, device=self.device, requires_grad=True)
        self.bias = torch.zeros(1, device=self.device, requires_grad=True)

        # Convert labels to 1-D
        X_t = torch.as_tensor(X, dtype=torch.float32,device=self.device)

        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)


        self.optimizer = torch.optim.SGD([self.weights, self.bias], lr=self.learning_rate)

        prev_loss = float('inf')
        for _ in range(self.num_iterations):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._sigmoid(linear_model)
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {self._loss(y_t, y_predicted).item()}")

            # loss = torch.nn.functional.binary_cross_entropy(y_predicted, y_t,reduction='mean')
            loss = self._loss(y_t, y_predicted)
            #reset the gradients to zero
            self.optimizer.zero_grad()            
            ## calculate gradients
            loss.backward()
            #update the weights and bias
            self.optimizer.step()

            # check for convergence
            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break   

        return self



    def predict(self, X,th=0.5):
        X = torch.tensor(X, dtype=torch.float32)
     
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
     

        linear_model = X @ self.weights + self.bias
        y_predicted = self._sigmoid(linear_model).detach().numpy()
        return np.array([1 if i > th else 0 for i in y_predicted])

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")
        
        linear_model = X @ self.weights + self.bias
        y_predicted = self._sigmoid(linear_model).detach().numpy()
        return y_predicted






#P(s =1 "| X")
# to be implemented labeling mechanism
def SAR(df,
    feature_col,
    label_col="Target",
    prob_func=None,
    random_state=42)-> pd.DataFrame:
    
    raise NotImplementedError("Please implement the SAR algorithm for your specific dataset.")





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

t = time.time()
print("Current time:", t)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
data = DataSets(name='BreastCancer')
X_train, y_train, X_test, y_test = data.get_X_y(test_size=0.2,
    c=1,labeling_mechanism="SCAR",
    train_balance=None,
    test_balance=None,
    scale_data="standard",
    random_state=42)
print("Data loaded in {:.2f} seconds".format(time.time() - t)   )
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)
print("First 5 rows of X_train:\n", X_train.head())  

clf = LogisticRegression()
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
report = classification_report(y_test.values, y_pred, output_dict=True)
print("training completed in {:.2f} seconds".format(time.time() - t))
print("Classification Report:")
print(pd.DataFrame(report).transpose())
cm = confusion_matrix(y_test.values, y_pred)    
print("Confusion Matrix:")
print(cm)

sk_clf = SklearnLogisticRegression(penalty=None,max_iter=1000)
sk_clf.fit(X_train.values, y_train.values)
y_pred_sk = sk_clf.predict(X_test.values)
report_sk = classification_report(y_test.values, y_pred_sk, output_dict=True)
print("Sklearn Classification Report:")
print(pd.DataFrame(report_sk).transpose())
cm_sk = confusion_matrix(y_test.values, y_pred_sk)    
print("Sklearn Confusion Matrix:")  
print(cm_sk)