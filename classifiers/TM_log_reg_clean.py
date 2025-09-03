import torch 
import numpy as np

from classifiers.helpers import _sigmoid,penalty,_loss
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import CONFIG
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class TwoModelLogReg_clean():
    def __init__(self,y_clf,e_clf,alpha=None,max_loop_iterations=None,epsilon=None):

        self.e = e_clf
        self.y = y_clf
        self.alpha = alpha 
        self.max_loop_iterations = max_loop_iterations if max_loop_iterations is not None else CONFIG.max_loop_iterations
        self.epsilon = epsilon if epsilon is not None else CONFIG.epsilon

        # super().__init__(lr=lr, max_iterations=max_iterations, tolerance=tolerance, _activation=_sigmoid, penalty=penalty, solver=solver)


    def fit(self,X,s):
        
        """
        Fits the model to the positive and unlabeled training data.
        """

        print("Starting TM fit...") 
        start = time.perf_counter()
        
        if type(X) != np.array:
            X = np.array(X)
        if type(s) != np.array:
            s = np.array(s)



        naive_clf = NaiveLogReg(max_iterations=300,penalty="l2",solver="adam")
        naive_clf.fit(X, s)
        
        alpha = naive_clf.get_c_hat() #get the c_hat from the naive classifier

        s_pred = naive_clf.predict_proba(X)
         #get the probabilities of the naive classifier
        e_pred = 1./2. * (s_pred + 1) #initial guess for e(x), see paper

        X_temp,s_temp,weights = self.calculate_weights(X_data=X,s_data=s,s_pred=s_pred,e_pred=e_pred)

        prev = np.inf

        for _ in range(self.max_loop_iterations):

            self.y.fit(X_temp, s_temp,sample_weight=weights)

            y_pred = self.predict_proba(X)

            y_pred_positive = y_pred[s == 1] #predictions for positive samples
            threshold = np.quantile(y_pred_positive, q=alpha) #calculate threshold based on alpha quantile


            p = self.pseudo_indices(y_pred,s, threshold)


            self.e.fit(X[p],s[p])

            e_pred = self.e.predict_proba(X)[:,1]

            s_pred = e_pred*y_pred

            X_temp,s_temp,weights = self.calculate_weights(X_data=X,s_data=s,s_pred=s_pred,e_pred=e_pred)

            loss = _loss(torch.as_tensor(s),torch.as_tensor(s_pred)).item() #spaghetti
            if  np.abs(loss - prev) < self.epsilon:
                print(f"TM Converged after {_} iterations with loss {loss:.4f}") 
                break

            prev = loss

        elapsed = time.perf_counter() - start
        print(f"TwoModelLogReg fitted in {elapsed:.4f} seconds ")

        self.y.fit(X_temp, s_temp,sample_weight=weights)

        return self

    def pseudo_indices(self, y_pred, s, threshold):
        """
        Define the pseudo-label set based on the current threshold.
        samples with predicted probabilities above the threshold are considered positive.

        return array of indices for possible positive samples
        """
        p = np.zeros(len(s), dtype=int)
        for idx,instance in enumerate(zip(y_pred, s)):
            if instance[1] == 1: #label
                p[idx] = 1
            else:
                if instance[0] > threshold: # 
                    p[idx] = 1
        
        p_indices = p > 0
        return p_indices









    def calculate_weights(self,X_data,s_data, s_pred, e_pred):
        """
        Calculate sample weights based on the current predictions.
        """

        #OddsRatio():
        #     """
        #     Estimates the odds ratio for a given sample.
        #     Is the ratio between the odds of sample being unlabeled among the positives versus the odds of a sample being unlabeled among the the complete set of both positives and negatives.
        #     """
        def OddsRatio(e,s):
                return ((1 - e) / e) * (s / (1 - s))

        # Weight function unlabeled class
        def w0(s,OR):
            return (1-s) * (1 - OR)

        #Weight function positive class
        def w1(s,OR):
            return s + (1-s)* OR

        X_temp_pos = X_data[s_data == 1] # positives
        X_temp_un = X_data[s_data == 0] # unlabeled

        s_pos = s_data[s_data == 1]
        s_un = s_data[s_data == 0]


        OR = OddsRatio(e_pred, s_pred)
        sample_weights = np.zeros(len(s_data))
        sample_weights[s_data == 1] = 1
        sample_weights[s_data == 0] = w0(s_data[s_data == 0], OR[s_data == 0]) #Unlabeled as negative
        temp_arr = np.ones(s_data[s_data == 0].shape[0])
        temp_arr = w1(s_data[s_data == 0], OR[s_data == 0])
        sample_weights = np.concatenate((sample_weights, temp_arr), axis=0) #Unlabeled as positive

        return np.concatenate((X_temp_pos, X_temp_un,X_temp_un), axis=0), np.concatenate((s_pos,s_un,s_un), axis=0), sample_weights


    def predict(self, X):
        return self.y.predict(X)

    def predict_proba(self,X):
        return self.y.predict_proba(X)[:,1]
    
    # def get_weights(self):
    #     if self.y is None:
    #         raise ValueError("Model has not been trained yet. Call fit() before get_weights().")
    #     return self.y.get_weights()


    # def get_e_weights(self):

    #     return self.e.get_weights()
    
    def _get_y_clf(self):
        if self.y is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_y_clf().")
        return self.y
    
    def _get_e_clf(self):
        if self.e is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_e_clf().")
        return self.e
