import torch 
import numpy as np

from classifiers.helpers import _loss, _sigmoid,penalty
from classifiers.base_log_reg import BaseLogReg
from config import CONFIG
import time
import logging

class ClassicLogReg(BaseLogReg):
    def __init__(self,lr=None, max_iterations=None, tolerance=None, penalty=None, solver=None):

        super().__init__(lr, max_iterations, tolerance, _sigmoid,penalty,solver)
        self.init()

    def init(self):
        """
        If a hyperparameter is not provided, use the default from the config.
        """

        if self.lr is None:
            self.lr = CONFIG.lr
        if self.max_iterations is None:
            self.max_iterations = CONFIG.max_iterations
        if self.tolerance is None:
            self.tolerance = CONFIG.tolerance
        if self.penalty is None:
            self.penalty = CONFIG.penalty
        if self.solver is None:
            self.solver = CONFIG.solver

        #Set random seed for reproducibility
        torch.manual_seed(CONFIG.random_state)
        np.random.seed(CONFIG.random_state)

    def fit(self,X,y):
        """
        Fit the model to the training data.
        lbfgs solver requries different training loop than adam
        call the fit_lbfgs method for lbfgs solver
        adam is more stable lbfgs but may converge faster
        l1 penalty is not supported with lbfgs solver, because l-bfgs assumes a smooth loss landscape.

        """

        start = time.perf_counter()
        if str(self.solver).lower() != 'lbfgs':
          result =  self.fit_adam(X,y)
        elif str(self.penalty).lower() == 'l1':
            raise ValueError("L1 penalty is not supported with LBFGS solver in this implementation.")
        else:
            result = self.fit_lbfgs(X,y)
        elapsed = time.perf_counter() - start
        logging.info(f"ClassicLogReg completed in {elapsed:.4f} seconds")
        return result

    def fit_adam(self, X, y):
        num_samples, n_features = X.shape

        self.weights = torch.zeros(n_features, device=CONFIG.device, requires_grad=True)
        self.bias = torch.zeros(1, device=CONFIG.device, requires_grad=True)

        X_t = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.device)

        self.optimizer = torch.optim.Adam([self.weights, self.bias],lr=self.lr)
        
        prev_loss = float('inf')

        self.loss_log = np.zeros(self.max_iterations)

        for _ in range(self.max_iterations):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._activation(linear_model)
            # if _ % 100 == 0:
            #     print(f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()}")

            loss = _loss(y_t, y_predicted)
            loss = penalty(self.penalty, loss, self.weights)

            self.optimizer.zero_grad() # reset grads
            loss.backward() # calculate grads
            self.optimizer.step() # update weights and bias

            self.loss_log[_] = loss.item() # log loss

            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break   
            prev_loss = loss.item()

        print(f"ClassicLogReg training complete, {_} iterations")
        return self

    def fit_lbfgs(self,X,y):
        num_samples, n_features = X.shape

        self.weights = torch.zeros(n_features, device=CONFIG.device, requires_grad=True)
        self.bias = torch.zeros(1, device=CONFIG.device, requires_grad=True)

        X_t = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.device)

        self.optimizer = torch.optim.LBFGS([self.weights, self.bias],lr=self.lr)

        prev_loss = float('inf')

        self.loss_log = np.zeros(self.max_iterations)

        for _ in range(self.max_iterations):
            def closure():
                self.optimizer.zero_grad()
                linear_model = X_t @ self.weights + self.bias
                y_predicted = self._activation(linear_model)
                loss = _loss(y_t, y_predicted)

                #Compute penalties if penalty has been set to l1 or l2
                loss = penalty(self.penalty, loss, self.weights)
                
                loss.backward()
                return loss
            
            self.optimizer.step(closure)
            
            #Compute logs:
            with torch.no_grad():
                linear_model = X_t @ self.weights + self.bias
                y_predicted = self._activation(linear_model)
                loss = _loss(y_t, y_predicted)
                loss = penalty(self.penalty, loss, self.weights)
                self.loss_log[_] = loss.item()

            # if _ % 100 == 0:
                # print(f"Iteration {_}, Loss: {loss.item()}")

            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"\n \n Converged after {_} iterations")
                break   

            prev_loss = loss.item()



        return self


    def predict(self, X, threshold=0.5):
        y_predict_proba = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_predict_proba])


    def predict_proba(self, X):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model).detach().numpy()


    def predict_torch_proba(self,X_t):
        with torch.no_grad():      
            linear_model = X_t @ self.weights + self.bias
            probs = self._activation(linear_model)
            return probs
        
