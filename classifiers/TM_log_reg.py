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
# from sklearn.linear_model import LogisticRegression

class TwoModelLogReg():
    def __init__(self,max_loop_iterations=None, epsilon=None, alpha=None,y_clf_args=None,e_clf_args=None, validation=None,verbose=1):

        self.max_loop_iterations = max_loop_iterations
        self.epsilon = epsilon
        self.alpha = alpha
        self.y_clf_args = y_clf_args
        self.e_clf_args = e_clf_args
        self.init()

        self.e = None
        self.y = None
        self.s = None
        self.OR = None
        self.iter = 0 #iteration counter
     
        self.val_log = []  
        self.VAL = [validation[0],validation[1],validation[2]] if validation is not None else None # validation set: X_val, y_val, s_val
        
        self.verbose = verbose
        # super().__init__(lr=lr, max_iterations=max_iterations, tolerance=tolerance, _activation=_sigmoid, penalty=penalty, solver=solver)
    
    def init(self):
        
        if self.max_loop_iterations is None:    
            self.max_loop_iterations = CONFIG.max_loop_iterations #Max outerloop iters
        if self.epsilon is None:
            self.epsilon = CONFIG.epsilon
        if self.alpha is None:
            self.alpha = CONFIG.alpha


    def init_e(self,clf_args=None):
        if clf_args is not None:
            assert type(clf_args) == dict, "e_clf_args must be a dictionary"
            
            if clf_args.get("clf") is not None:
                return clf_args.get("clf")
        else:
            clf_args = {}   

        lr = clf_args.get("lr") if clf_args.get("lr") is not None else CONFIG.lr
        max_iterations = clf_args.get("max_iterations") if clf_args.get("max_iterations") is not None else CONFIG.max_iterations
        tolerance = clf_args.get("tolerance") if clf_args.get("tolerance") is not None else CONFIG.tolerance
        penalty = clf_args.get("penalty") if clf_args.get("penalty") is not None else CONFIG.penalty
        solver = clf_args.get("solver") if clf_args.get("solver") is not None else CONFIG.solver
        return ClassicLogReg(lr=lr, max_iterations=max_iterations, tolerance=tolerance, penalty=penalty, solver=solver)

    def init_y(self,clf_args=None):
        if clf_args is not None:
            assert type(clf_args) == dict, "e_clf_args must be a dictionary"
        else:
            clf_args = {}

        if clf_args.get("clf") is not None:
            return clf_args.get("clf")

        lr = clf_args.get("lr") if clf_args.get("lr") is not None else CONFIG.lr
        max_iterations = clf_args.get("max_iterations") if clf_args.get("max_iterations") is not None else CONFIG.max_iterations
        tolerance = clf_args.get("tolerance") if clf_args.get("tolerance") is not None else CONFIG.tolerance
        penalty = clf_args.get("penalty") if clf_args.get("penalty") is not None else CONFIG.penalty
        solver = clf_args.get("solver") if clf_args.get("solver") is not None else CONFIG.solver
        return self.Y(self,lr=lr, max_iterations=max_iterations, tolerance=tolerance, penalty=penalty, solver=solver)


    def fit(self,X,s):
        
        """
        Fits the model to the positive and unlabeled training data.
        """

    


        print("Starting TM fit...") if self.verbose == 1 else None
        #initialize 
        start = time.perf_counter()
        self.iter = 0
        # Fit (and validate) the naive classifier
        naive_clf = NaiveLogReg(max_iterations=300,penalty="l2",solver="adam")
        naive_clf.fit(X, s)

        if self.VAL is not None:
            self.validate(naive_clf,name="y(X)")

        if self.alpha is None:
            self.alpha = naive_clf.get_c_hat() #get the c_hat from the naive classifier

        s_pred = naive_clf.predict_torch_label_proba(X)
         #get the probabilities of the naive classifier
        e_pred = 1./2. * (s_pred + 1) #initial guess for e(x), see paper
        OR = self.OddsRatio(e_pred,s_pred)

        # self.e =ClassicLogReg(lr=self.lr, max_iterations=self.max_iterations, tolerance=self.tolerance, penalty=self.penalty, solver=self.solver)
        self.e = self.init_e(self.e_clf_args)
        self.y = self.init_y(self.y_clf_args)


        X = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.device)
        s = torch.as_tensor(s, dtype=torch.float32, device=CONFIG.device)
        prev = 0

        while self.iter < self.max_loop_iterations:

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

            # if self.iter % 100 == 0:
            #     print(f"Iteration {self.iter}")

            if self.VAL is not None:
                self.validate(self.y,name="y(X)")
                self.validate(self.e,name="e(X)",label_freq=True)
                self.val_log.append(("threshold",threshold.detach().numpy(),0,0,0))
                or_ = np.mean(self.OddsRatio(e=e_pred,s=s_pred).detach().numpy())
                self.val_log.append(("OR",or_,0,0,0))
                self.val_log.append(("size_p",np.sum(p.detach().numpy()),0,0,0))
            


            if  np.abs(loss - prev) < self.epsilon:
                logging.info(f"TM Converged after {self.iter} iterations with loss {loss:.4f}")
                print(f"TM Converged after {self.iter} iterations with loss {loss:.4f}") if self.verbose == 1 else None
                break

            prev = loss

        elapsed = time.perf_counter() - start
        logging.info(f"TwoModelLogReg fitted in {elapsed:.4f} seconds ")
        print(f"TwoModelLogReg fitted in {elapsed:.4f} seconds ") if self.verbose == 1 else None

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
        eps = 1e-7
        e = e.detach().clamp(eps, 1-eps)
        s = s.detach().clamp(eps, 1-eps)
        return ((1 - e) / e) * (s / (1 - s))


    class Y(ClassicLogReg):
        """
        Adaptation of the classic logistic regression with weight adjustment, can be trained on positive and unlabeled data.
        Weights are based on the odds ratio 
        """




        def __init__(self,out, lr=None, max_iterations=None, tolerance=None, penalty=None, solver=None):
            super().__init__(lr, max_iterations, tolerance, penalty, solver)
            self.out = out

        # Weight function unlabeled class
        def w0(self,s,OR):
            return (1-s) + s * OR

        #Weight function positive class
        def w1(self,s,OR):
            return s + (1-s)* OR

        # Loss is adjusted based on class of samples
        def _weighted_loss(self,s,s_pred,OR):
            eps = 1e-7
            s_pred = s_pred.clamp(eps, 1 - eps)  # clamp to avoid log(0)
            W = self.w1(s, OR)* torch.log(s_pred) + self.w0(s, OR)* torch.log(1. - s_pred)
            return -torch.mean(W) # clamp to avoid log(0)


        #Adaptation of the classic logistic regression fit function with adam solver
        # Loss is now with weight adjustment
        def fit(self, X, s, OR):

            # OR = OR.detach()
            
            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.device, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.device, requires_grad=True)

            self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.lr)
            
            prev_loss = float('inf')

            self.loss_log = np.zeros(self.max_iterations)

            for _ in range(self.max_iterations):
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

                if abs(prev_loss - loss.item()) < self.tolerance:
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
