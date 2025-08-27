import torch 
import numpy as np

from classifiers.helpers import _sigmoid,penalty,_loss
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import CONFIG
import copy
import time
import logging
# from sklearn.linear_model import LogisticRegression

class TwoModelLogReg(BaseLogReg):
    def __init__(self, learning_rate=CONFIG.LEARNING_RATE, epochs=300, tolerance=CONFIG.CONVERGENCE_TOLERANCE, penalty=CONFIG.penalty, solver=CONFIG.solver, alpha=CONFIG.TM_ALPHA, epsilon=CONFIG.CONVERGENCE_TOLERANCE, validation=None):

        super().__init__(learning_rate, epochs, tolerance, _sigmoid,penalty,solver)
        self.naive_clf = NaiveLogReg(epochs=300,penalty="l2",solver="adam")
        self.e = None
        self.y = None
        self.s = None
        self.OR = None
        self.iter = 0 #iteration counter
        self.alpha = alpha #alpha is the quantile for the threshold 
        self.epsilon = epsilon
        self.val_log = []  
        self.VAL = [validation[0],validation[1],validation[2]] if validation is not None else None # validation set: X_val, y_val, s_val
   
    def fit(self,X,s):
        
        """
        Fits the model to the positive and unlabeled training data.
        """

    



        #initialize 
        start = time.perf_counter()
        self.iter = 0
        # Fit (and validate) the naive classifier
        self.naive_clf.fit(X, s)
        if self.VAL is not None:
            self.validate(self.naive_clf,name="y(X)")

        if self.alpha is None:
            self.alpha = self.naive_clf.get_c_hat() #get the c_hat from the naive classifier
        s_pred = self.naive_clf.predict_torch_label_proba(X)
         #get the probabilities of the naive classifier
        e_pred = 1./2. * (s_pred + 1) #initial guess for e(x), see paper
        OR = self.OddsRatio(e_pred,s_pred)

        self.e =ClassicLogReg(learning_rate=self.learning_rate*0.5, epochs=600,penalty="l1",solver="adam")
        self.y = self.Y(out=self, learning_rate=self.learning_rate, epochs=self.epochs,penalty=self.penalty,solver=self.solver)


        X = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.TORCH_DEVICE)
        s = torch.as_tensor(s, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)
        prev = None

        while self.iter < self.epochs:

            loss = self.y.fit(X, s,OR)

            y_pred = self.y.predict_torch_proba(X) 

            y_pred_positive = y_pred[s == 1] #predictions for positive samples
            threshold = torch.quantile(y_pred_positive, self.alpha) #calculate threshold based on alpha quantile


            p = self.pseudo_indices(y_pred,s, threshold)

            # self.alpha = torch.as_tensor(len(p) / len(s),dtype=torch.float32) #update alpha based on the current pseudo-labels

            self.e.fit(X[p],s[p])

            #sklearn:
            # e_pred = torch.as_tensor(self.e.predict_proba(X)[:,1], dtype=torch.float32)

            e_pred = self.e.predict_torch_proba(X)

            s_pred = e_pred.detach()*y_pred.detach()
            
            OR = self.OddsRatio(e_pred,s_pred)


            self.iter += 1

            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}")

            if self.VAL is not None:
                self.validate(self.y,name="y(X)")
                self.validate(self.e,name="e(X)",label_freq=True)
                self.val_log.append(("threshold",threshold.detach().numpy(),0,0,0))
                or_ = np.mean(self.OddsRatio(e=e_pred,s=s_pred).detach().numpy())
                self.val_log.append(("OR",or_,0,0,0))
                self.val_log.append(("size_p",np.sum(p.detach().numpy()),0,0,0))
            


            if ( prev is not None ) and  ( np.abs(loss - prev) < self.epsilon ):
                logging.info(f"Converged after {self.iter} iterations with loss {loss:.4f}")
                print(f"Converged after {self.iter} iterations with loss {loss:.4f}")
                break

            prev = loss

        elapsed = time.perf_counter() - start
        logging.info(f"TwoModelLogReg completed in {elapsed:.4f} seconds")
        return self

    def pseudo_indices(self, y_pred, s, threshold):
        """
        Define the pseudo-label set based on the current threshold.
        samples with predicted probabilities above the threshold are considered positive.

        return array of indices for possible positive samples
        """
        p = torch.zeros(len(s), dtype=int)
        for idx,instance in enumerate(zip(y_pred, s)):
            if instance[1] == 1: #label
                p[idx] = 1
            else:
                if instance[0] > threshold: # 
                    p[idx] = 1
        
        p_indices = p > 0
        return p_indices






    #OddsRatio():
    #     """
    #     Estimates the odds ratio for a given sample.
    #     Is the ratio between the odds of sample being unlabeled among the positives versus the odds of a sample being unlabeled among the the complete set of both positives and negatives.

    @staticmethod
    def OddsRatio(e,s):
        eps = 1e-8
        e = e.detach().clamp(eps, 1-eps)
        s = s.detach().clamp(eps, 1-eps)
        return ((1 - e) / e) * (s / (1 - s))


    class Y(ClassicLogReg):
        """
        Adaptation of the classic logistic regression with weight adjustment, can be trained on positive and unlabeled data.
        Weights are based on the odds ratio 
        """

        def __init__(self,out, learning_rate=0.001, epochs=100, tolerance=0.001, penalty="l2", solver='lbfgs'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)
            self.out = out

        # Weight function unlabeled class
        def w0(self,s,OR):
            return (1-s) + s * OR

        #Weight function positive class
        def w1(self,s,OR):
            return s + (1-s)* OR

        # Loss is adjusted based on class of samples
        def _weighted_loss(self,s,s_pred,OR):
            eps = 1e-8
            s_pred = s_pred.clamp(eps, 1 - eps)  # clamp to avoid log(0)
            W = self.w1(s, OR)* torch.log(s_pred) + self.w0(s, OR)* torch.log(1. - s_pred)
            return -torch.mean(W) # clamp to avoid log(0)


        #Adaptation of the classic logistic regression fit function with adam solver
        # Loss is now with weight adjustment
        def fit(self, X, s, OR):

            # OR = OR.detach()
            
            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

            self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
            
            prev_loss = float('inf')

            self.loss_log = np.zeros(self.epochs)

            for _ in range(self.epochs):
                linear_model = X @ self.weights + self.bias
                s_pred = self._activation(linear_model)
                # if _ % 100 == 0:
                #     print(f"Iteration {_}, Loss: {self._weighted_loss(s_t, y_predicted,X).item()}")

                loss = self._weighted_loss(s=s,s_pred=s_pred,OR=OR)
                loss = penalty(self.penalty, loss, self.weights)

                self.optimizer.zero_grad() # reset grads
                loss.backward() # calculate grads
                self.optimizer.step() # update weights and bias

                self.loss_log[_] = loss.item() # log loss

                if abs(prev_loss - loss.item()) < self.tolerance or loss.item() < self.tolerance:
                    # print(f"Converged after {_} iterations")
                    break  
                
                prev_loss = loss.item()
            return loss.item()


    
    # VALDIDATION ###########################################################


    def validate(self,clf,name,label_freq=False):
        X_val, y_val, s_val = self.VAL

        if label_freq:
            X_val = X_val[y_val == 1]
            s_val = s_val[y_val == 1]
            y_true = s_val
        else:
            y_true = y_val

        y_pred = clf.predict(X_val)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=1)
        recall = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        self.val_log.append((name,accuracy, precision, recall, f1))


        # logging.info(f"{name} - - Validation at iteration {clf.iter}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

    #########################################################################


    def predict(self, X, threshold=0.5):
        return self.y.predict(X, threshold)
    
    def predict_proba(self,X):
        return self.y.predict_proba(X)
    
    def get_weights(self):
        if self.y is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_weights().")
        return self.y.get_weights()


    def get_e_weights(self):

        return self.e.get_weights()
    
    def _get_y_clf(self):
        if self.y is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_y_clf().")
        return self.y
    
    def _get_e_clf(self):
        if self.e is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_e_clf().")
        return self.e

    def get_validation_logs(self):
        if self.VAL is None:
            raise ValueError("Validation set is not available. Make sure to provide a validation set during initialization.")
        else:
            return self.val_log

    def validate2log(self):
        if self.VAL is None:
            raise ValueError("Validation set is not available. Make sure to provide a validation set during initialization.")
        else:
            for i,log in enumerate(self.val_log):
                logging.info(f"Validation log {i}: {log}")
