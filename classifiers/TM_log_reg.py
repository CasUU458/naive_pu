import torch 
import numpy as np

from classifiers.helpers import _sigmoid,penalty
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from config import CONFIG
import time
import logging

class TwoModelLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=1000, tolerance=1e-6,penalty=None,solver='adam',alpha=0.5,epsilon=1e-8):

        super().__init__(learning_rate, epochs, tolerance, _sigmoid,penalty,solver)
        self.naive_clf = NaiveLogReg(epochs=300,penalty="l2",solver="adam")
        self.e = None
        self.y = None
        self.s = None
        self.OR = None
        self.iter = 0 #iteration counter
        self.alpha = alpha #alpha is the quantile for the threshold 
        self.epsilon = epsilon

    def fit(self,X,y):
        
        """
        Fit the model to the training data.
        """
        start = time.perf_counter()
        self.iter = 0
        self.naive_clf.fit(X, y)
        self.e = self.E()
        self.y = self.Y()
        self.s = self.S(self, self.e, self.y)
        self.OR = self.OddsRatio(self.e, self.s)

        while self.iter < self.epochs and converge_treshold > self.epsilon:
            self.y.fit(X, y)
            threshold = self.calc_threshold(X,alpha=self.alpha)
            p = self.define_psuedo_set(X, threshold)
            self.e.fit(X, p)

            self.s.update(self.e, self.y)
            self.OR.update(self.e, self.s)

            self.iter += 1
            converge_treshold = 1
        

        elapsed = time.perf_counter() - start
        logging.info(f"ClassicLogReg completed in {elapsed:.4f} seconds")
        return self

    def define_psuedo_set(self, X,y, threshold):
        """
        Define the pseudo-label set based on the current threshold.
        samples with predicted probabilities above the threshold are considered positive.

        return array of indices for possible positive samples
        """
        p = np.zeros(len(X), dtype=int)
        for i,instance in enumerate(zip(X, y)):
            if instance[1] == 1:
                p[i] = 1
            else:
                if self.y.predict_proba(instance[0]) > threshold:
                    p[i] = 1
        return p
    


    def calc_threshold(self,x):
        return self.quantile(x)

    def quantile(self, X):
        probs = self.y.predict_proba(X)
        return np.quantile(probs, self.alpha)

    class S():
        def __init__(self, out, e, y):
            self.out = out
            self.e = e.copy()
            self.y = y.copy()
        
        def __call__(self, X):
            
            if self.out.iter > 0:
                f = self.y.predict_proba(X)*self.e.predict_proba(X)
                return f
            else:
                return self.s_naive(X)

        def s_naive(self,X):
            return self.out.naive_clf.predict_label_proba(X)

        def update(self,e,y):
            self.e = e.copy()
            self.y = y.copy()

    class OddsRatio():
        def __init__(self, e,s):
            self.e = e.copy()
            self.s = s.copy()

        def __call__(self, X):
            e = self.e.predict_proba(X)
            s = self.s(X)
            return (e / (1 - e)) * ((1-s) / s)

        def update(self,e,s):
            self.e = e.copy()
            self.s = s.copy()

    class Y(ClassicLogReg):
        def __init__(self,out, learning_rate=0.001, epochs=1000, tolerance=0.000001, penalty=None, solver='adam'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)


        def w0(self,s,X):
            return (1-s) +s*self.out.OR(X)
        
        def w1(self,s,X):
            return s+(1-s)*self.out.OR(X)

        def _weighted_loss(self,s,y_pred,X):
            eps = 1e-15  # to avoid log(0), numerical stability, small constant
            # ensure y_pred is in the range [eps, 1-eps]
            #clamp_min y_pred to avoid log(0)

            s = s.clamp(eps, 1. - eps)
            W = self.w1(s, X)* torch.log(y_pred) + self.w0(s, X)* torch.log(1. - y_pred)
            return -torch.mean(W)      
        

        
        
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
                if _ % 100 == 0:
                    print(f"Iteration {_}, Loss: {self._weighted_loss(s_t, y_predicted,X).item()}")

                loss = self._weighted_loss(s_t, y_predicted,X)
                loss = penalty(self.penalty, loss, self.weights)

                self.optimizer.zero_grad() # reset grads
                loss.backward() # calculate grads
                self.optimizer.step() # update weights and bias

                self.loss_log[_] = loss.item() # log loss

                if abs(prev_loss - loss.item()) < self.tolerance:
                    print(f"Converged after {_} iterations")
                    break  
                
                prev_loss = loss.item()
            return self

    class E(ClassicLogReg):
        def __init__(self,out, learning_rate=0.001, epochs=1000, tolerance=0.000001, penalty=None, solver='adam'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)
            self.out = out

        def predict_proba(self, X):
            if self.out.iter > 0:
                linear_model = self.update_linear_model(X)
                return self._activation(linear_model).detach().numpy()
            #Polynomial estimate of e, see paper 
            return 1/2*(self.s_naive(X) + 1)


    def predict(self, X, threshold=0.5):
        return self.y.predict(X, threshold)
    
    def predict_proba(self,X):
        return self.y.predict_proba(X)