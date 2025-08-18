import torch 
import numpy as np

from classifiers.helpers import _sigmoid,penalty
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from config import CONFIG
import copy
import time
import logging

class TwoModelLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=300, tolerance=1e-6,penalty=None,solver='adam',alpha=0.5,epsilon=1e-8):

        super().__init__(learning_rate, epochs, tolerance, _sigmoid,penalty,solver)
        self.naive_clf = NaiveLogReg(epochs=300,penalty="l2",solver="adam")
        self.e = None
        self.y = None
        self.s = None
        self.OR = None
        self.iter = 0 #iteration counter
        self.alpha = alpha #alpha is the quantile for the threshold 
        self.epsilon = epsilon

    def fit(self,X,s):
        
        """
        Fits the model to the positive and unlabeled training data.
        """

        #initialize 
        start = time.perf_counter()
        self.iter = 0
        self.naive_clf.fit(X, s)
        self.e = self.E(out=self)
        self.y = self.Y(out=self)
        self.s = self.S(self, self.e, self.y)
        self.OR = self.OddsRatio(self.e, self.s)

        converge_treshold = 1

        while self.iter < self.epochs:

            self.y.fit(X, s)


            pred = self.y.predict_proba(X) 
            threshold = self.calc_threshold(pred)
            p = self.define_psuedo_set(X,s, threshold)


            self.e.fit(X[p],s[p])

            # self.s.update(self.e, self.y)
            # self.OR.update(self.e, self.s)

            self.iter += 1
            converge_treshold = 1

            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}")

        elapsed = time.perf_counter() - start
        logging.info(f"ClassicLogReg completed in {elapsed:.4f} seconds")
        return self

    def define_psuedo_set(self, X, s, threshold):
        """
        Define the pseudo-label set based on the current threshold.
        samples with predicted probabilities above the threshold are considered positive.

        return array of indices for possible positive samples
        """
        p = np.zeros(len(X), dtype=int)
        for idx,instance in enumerate(zip(X, s)):
            if instance[1] == 1: #label
                p[idx] = 1
            else:
                if self.y.predict_proba(instance[0]) > threshold: # 
                    p[idx] = 1
        
        p_indices = p > 0
        return p_indices


    def calc_threshold(self,pred):
        
        pred = np.sort(pred)
        

        return self.quantile(pred)

    def quantile(self, X):
        return np.quantile(X, self.alpha)

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

        def __call__(self, X):
            
            if self.out.iter > 0:
                f = self.y.predict_proba(X)*self.e.predict_proba(X)
                return f
            else:
                return self.s_naive(X)

        def s_naive(self,X):
            return self.out.naive_clf.predict_label_proba(X)

        # def update(self,e,y):
        #     self.e = copy.deepcopy(e)
        #     self.y = copy.deepcopy(y)

    class OddsRatio():
        """
        Estimates the odds ratio for a given sample.
        Is the ratio between the odds of sample being unlabeled among the positives versus the odds of a sample being unlabeled among the the complete set of both positives and negatives.

        """

        def __init__(self, e,s):
            self.e = e
            self.s = s

        def __call__(self, X):
            e = self.e.predict_proba(X)
            s = self.s(X)
            return (e / (1 - e)) * ((1-s) / s)

        # def update(self,e,s):
        #     self.e = copy.deepcopy(e)
        #     self.s = copy.deepcopy(s)

    class Y(ClassicLogReg):
        """
        Adaptation of the classic logistic regression with weight adjustment, can be trained on positive and unlabeled data.
        Weights are based on the odds ratio 
        """

        def __init__(self,out, learning_rate=0.001, epochs=300, tolerance=0.01, penalty="l2", solver='adam'):
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

            X_t = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.TORCH_DEVICE)
            s_t = torch.as_tensor(s, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)

            self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
            
            prev_loss = float('inf')

            self.loss_log = np.zeros(self.epochs)

            for _ in range(self.epochs):
                linear_model = X_t @ self.weights + self.bias
                y_predicted = self._activation(linear_model)
                # if _ % 100 == 0:
                #     print(f"Iteration {_}, Loss: {self._weighted_loss(s_t, y_predicted,X).item()}")

                loss = self._weighted_loss(s_t, y_predicted,X)
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
        def __init__(self,out, learning_rate=0.001, epochs=300, tolerance=0.01, penalty="l2", solver='adam'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)
            self.out = out


        #Adjusted prediction versus classic logreg 
        #
        def predict_proba(self, X):
            #Initial guess requried for algorithm
            if self.weights is None or self.bias is None:
                # print("Model is not trained yet, make initial guess based navie pu log reg")
                return 1/2*(self.out.s.s_naive(X) + 1)
            
            linear_model = self.update_linear_model(X)
            return self._activation(linear_model).detach().numpy()
            #Polynomial estimate of e, see paper 
            


    def predict(self, X, threshold=0.5):
        return self.y.predict(X, threshold)
    
    def predict_proba(self,X):
        return self.y.predict_proba(X)
    

# TEST the algorithm

