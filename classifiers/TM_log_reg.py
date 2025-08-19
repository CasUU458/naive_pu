import torch 
import numpy as np

from classifiers.helpers import _sigmoid,penalty
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import CONFIG
import copy
import time
import logging

class TwoModelLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=300, tolerance=1e-6,penalty=None,solver='adam',alpha=0.5,epsilon=1e-8,validation=None):

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

        self.e = self.E(out=self)
        self.y = self.Y(out=self, learning_rate=self.learning_rate, epochs=self.epochs,penalty=self.penalty,solver=self.solver)
        self.s = self.S(out=self, e=self.e, y=self.y)
        self.OR = self.OddsRatio(out=self,e=self.e, s=self.s)
   

        converge_treshold = 1
        X = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.TORCH_DEVICE)
        s = torch.as_tensor(s, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)


        while self.iter < self.epochs:

            self.y.fit(X, s)

            pred = self.y.predict_torch_proba(X) 
            threshold = self.calc_threshold(pred)
            p = self.define_psuedo_set(X[p],s[p], threshold)


            self.e.fit(X,s)

            self.iter += 1
            converge_treshold = 1

            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}")

            if self.VAL is not None:
                self.validate(self.y,name="y(X)")
                self.validate(self.e,name="e(X)",label_freq=True)
                self.val_log.append(("threshold",threshold.detach().numpy(),0,0,0))
                or_ = np.mean(self.OR(torch.as_tensor(X, dtype=torch.float32)).detach().numpy())
                self.val_log.append(("OR",or_,0,0,0))


        elapsed = time.perf_counter() - start
        logging.info(f"TwoModelLogReg completed in {elapsed:.4f} seconds")
        return self

    def define_psuedo_set(self, X, s, threshold):
        """
        Define the pseudo-label set based on the current threshold.
        samples with predicted probabilities above the threshold are considered positive.

        return array of indices for possible positive samples
        """
        p = torch.zeros(len(X), dtype=int)
        for idx,instance in enumerate(zip(X, s)):
            if instance[1] == 1: #label
                p[idx] = 1
            else:
                if self.y.predict_torch_proba(instance[0]) > threshold: # 
                    p[idx] = 1
        
        p_indices = p > 0
        return p_indices


    def calc_threshold(self,pred):
        
        threshold = torch.quantile(pred, self.alpha)
    

        return threshold


    class S():
        """
        Non traditional clf, gives the probability that a given sample is labeled or not. 
        
        s(x) = e(x)*y(x)
        s(x) = P(s= 1|x) = P(s = 1|,y = 1,x) * P(y=1|x)
    
        """
        

        def __init__(self, out, e, y):
            self.out = out
            self.e = e
            self.y = y

        def __call__(self,X):
            if self.e.weights is None or self.y.weights is None:
                # print("Model is not trained yet, make initial guess based navie pu log reg")
                return self.s_naive(X)               
            return self.e.predict_torch_proba(X)*self.y.predict_torch_proba(X)

        def s_naive(self,X):
            return self.out.naive_clf.predict_torch_label_proba(X)


    class OddsRatio():
        """
        Estimates the odds ratio for a given sample.
        Is the ratio between the odds of sample being unlabeled among the positives versus the odds of a sample being unlabeled among the the complete set of both positives and negatives.

        """

        def __init__(self,out, e,s):
            self.e = e
            self.s = s
            self.out = out
       
        def __call__(self, X):
            e = self.e.predict_torch_proba(X)
            s = self.s(X)
            return (e * (1 - s)) / ((1 - e) * s)



    class Y(ClassicLogReg):
        """
        Adaptation of the classic logistic regression with weight adjustment, can be trained on positive and unlabeled data.
        Weights are based on the odds ratio 
        """

        def __init__(self,out, learning_rate=0.001, epochs=100, tolerance=0.001, penalty="l2", solver='lbfgs'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)
            self.out = out

        # Weight function unlabeled class
        def w0(self,s,X):
            return (1-s) +s*self.out.OR(X)

        #Weight function positive class
        def w1(self,s,X):
            return s+(1-s)*self.out.OR(X)

        # Loss is adjusted based on class of samples
        def _weighted_loss(self,s,y_pred,X):
            eps = 1e-15  # to avoid log(0), numerical stability, small constant
            # ensure y_pred is in the range [eps, 1-eps]
            #clamp_min y_pred to avoid log(0)

            s = s.clamp(eps, 1. - eps)
            W = self.w1(s, X)* torch.log(y_pred) + self.w0(s, X)* torch.log(1. - y_pred)
            return -torch.mean(W)      
        


        #Adaptation of the classic logistic regression fit function with adam solver
        # Loss is now with weight adjustment
        def fit(self, X, s):
            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

            self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
            
            prev_loss = float('inf')

            self.loss_log = np.zeros(self.epochs)

            for _ in range(self.epochs):
                linear_model = X @ self.weights + self.bias
                y_predicted = self._activation(linear_model)
                # if _ % 100 == 0:
                #     print(f"Iteration {_}, Loss: {self._weighted_loss(s_t, y_predicted,X).item()}")

                loss = self._weighted_loss(s, y_predicted,X)
                loss = penalty(self.penalty, loss, self.weights)

                self.optimizer.zero_grad() # reset grads
                loss.backward() # calculate grads
                self.optimizer.step() # update weights and bias

                self.loss_log[_] = loss.item() # log loss

                if abs(prev_loss - loss.item()) < self.tolerance:
                    # print(f"Converged after {_} iterations")
                    break  
                
                prev_loss = loss.item()
            return self

    class E(ClassicLogReg):
        def __init__(self,out, learning_rate=0.001, epochs=100, tolerance=0.001, penalty="l2", solver='adam'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)
            self.out = out


        #Adjusted prediction versus classic logreg 
        #
        def predict_proba(self, X):
            #Initial guess requried for algorithm
            if self.weights is None:
                # print("Model is not trained yet, make initial guess based navie pu log reg")
                return 1/2*(self.out.s.s_naive(X).detach().numpy() + 1)
            
            linear_model = self.update_linear_model(X)
            return self._activation(linear_model).detach().numpy()
            #Polynomial estimate of e, see paper 

        def predict_torch_proba(self, X_t):
            if self.weights is None:
                # print("Model is not trained yet, make initial guess based navie pu log reg")
                return 1/2*(self.out.s.s_naive(X_t) + 1)
            return super().predict_torch_proba(X_t)
    
    # VALDIDATION ###########################################################


    def validate(self,clf,name,label_freq=False):
        X_val, y_val, s_val = self.VAL

        if label_freq:
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
