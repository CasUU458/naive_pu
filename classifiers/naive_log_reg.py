import torch
import numpy as np

from classifiers.helpers import _loss,penalty
from classifiers.base_log_reg import BaseLogReg
from classifiers.helpers import c2b, _modified_pu_sigmoid, b2c
from config import CONFIG
import time
import logging


class NaiveLogReg(BaseLogReg):
    def __init__(self,max_iterations=None, lr=None,lr_c=None, tolerance=None, c_estimate=None, penalty=None, solver=None,random_state=None):
        

        self.optimizer_b = None
        self.b = None #b parameter surrogate for c in modified sigmoid function
        self.c_log = None
        self.loss_c_log = None

        self.lr = lr
        self.lr_c = lr_c
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.c_estimate = c_estimate
        self.penalty = penalty
        self.solver = solver
        self.random_state = random_state
        self.init() #use value from config if provided parameter value is none
        super().__init__(lr=self.lr, max_iterations=self.max_iterations, tolerance=self.tolerance, _activation=_modified_pu_sigmoid, penalty=self.penalty, solver=self.solver)


       

        
        
        
        
        
        
        


        
    def init(self):
        """
        Imports values from the config file if parameters values are not provided by user
        """


        if self.c_estimate is None:
            self.b_init = None
        else:
            self.b_init = c2b(self.c_estimate)

        if self.lr is None:
            self.lr = CONFIG.lr

        if self.lr_c is None:
            self.lr_c = CONFIG.lr_c

        if self.max_iterations is None:
            self.max_iterations = CONFIG.max_iterations

        if self.tolerance is None:
            self.tolerance = CONFIG.tolerance

        if self.penalty is None:
            self.penalty = CONFIG.penalty

        if self.solver is None:
            self.solver = CONFIG.solver

        if self.random_state is None:
            self.random_state = CONFIG.random_state
        return 0

    def fit(self, X, y):
        """
        Fit the model to the training data.
        lbfgs solver requries different training loop than adam
        call the fit_lbfgs method for lbfgs solver
        adam is more stable lbfgs but may converge faster
        l1 penalty is not supported with lbfgs solver, because l-bfgs assumes a smooth loss landscape.

        """
    
        if self.b_init is None:
            self.b_init = c2b(np.random.default_rng(CONFIG.random_state).uniform(0.2, 0.8)) # avoid b being None if c init is None


        start = time.perf_counter()
        if str(self.solver).lower() != 'lbfgs':
            result =  self.fit_adam(X,y)
        elif str(self.penalty).lower() == 'l1':
            raise ValueError("L1 penalty is not supported with LBFGS solver in this implementation.")
        else:
            result = self.fit_lbfgs(X,y)
        elapsed = time.perf_counter() - start
        logging.info(f"NaiveLogReg completed in {elapsed:.4f} seconds")
        return result

    def fit_adam(self, X, y):
        num_samples, n_features = X.shape


        self.weights = torch.zeros(n_features, device=CONFIG.device, requires_grad=True)
        self.bias = torch.zeros(1, device=CONFIG.device, requires_grad=True)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=CONFIG.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.device)

        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.lr)

        self.b = torch.nn.Parameter(torch.tensor(0.5, device=CONFIG.device, requires_grad=True))
        self.optimizer_b = torch.optim.Adam([self.b], lr=self.lr_c)

        prev_loss = float('inf')
        self.loss_log, self.loss_c_log, self.c_log = np.zeros(self.max_iterations), np.zeros(self.max_iterations), np.zeros(self.max_iterations)

        for _ in range(self.max_iterations):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._activation(linear_model, self.b)

            loss = _loss(y_t, y_predicted)
            loss = penalty(self.penalty, loss, self.weights)

            self.optimizer.zero_grad() # reset grads
            loss.backward() # calculate grads
            self.optimizer.step() # update weights and bias

            self.loss_log[_] = loss.item()

            # NAIVE B OPTIMIZATION
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._activation(linear_model, self.b)

            loss_b = _loss(y_t, y_predicted)
            
            self.optimizer_b.zero_grad() # reset grads
            loss_b.backward() # calculate grads
            self.optimizer_b.step() # update b

            self.loss_c_log[_] = loss_b.item()
            self.c_log[_] = b2c(self.b.item())

            if _ % 1000 == 0:
                print(
                    f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()} Loss b: {loss_b.item()}, c: {b2c(self.b.item())}")

            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break

            prev_loss = loss.item()
        return self

    def fit_lbfgs(self, X, y):
            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.device, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.device, requires_grad=True)

            X_t = torch.as_tensor(X, dtype=torch.float32, device=CONFIG.device)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.device)

            self.optimizer = torch.optim.LBFGS([self.weights, self.bias], lr=self.lr)

            self.b = torch.nn.Parameter(torch.tensor(float(self.b_init), dtype=torch.float32, device=CONFIG.device))
            self.optimizer_b = torch.optim.LBFGS([self.b], lr=self.lr_c)

            prev_loss = float('inf')
            self.loss_log, self.loss_c_log, self.c_log = np.zeros(self.max_iterations), np.zeros(self.max_iterations), np.zeros(self.max_iterations)

            
            for _ in range(self.max_iterations):
                def closure():
                    self.optimizer.zero_grad()
                    linear_model = X_t @ self.weights + self.bias
                    y_predicted = self._activation(linear_model, self.b)
                    loss = _loss(y_t, y_predicted)

                    #Compute penalties if penalty has been set to l1 or l2
                    if self.penalty is not None:
                        loss = penalty(self.penalty, loss, self.weights)

                    loss.backward()
                    return loss
                
                self.optimizer.step(closure)
                loss = closure()
                def closure_b():
                    self.optimizer_b.zero_grad()
                    linear_model = X_t @ self.weights + self.bias
                    y_predicted = self._activation(linear_model, self.b)
                    loss = _loss(y_t, y_predicted)

                    #Compute penalties if penalty has been set to l1 or l2
                    if self.penalty is not None:
                        loss = penalty(self.penalty, loss, self.weights)

                    loss.backward()
                    return loss

                self.optimizer_b.step(closure_b)
                loss_b = closure_b()

                self.loss_log[_] = loss.item()
                self.loss_c_log[_] = loss_b.item()
                self.c_log[_] = b2c(self.b.item())

                if _ % 100 == 0:
                    print(f"Iteration {_}, Loss: {loss.item()}, Loss b: {loss_b.item()}, c: {b2c(self.b.item())}")

                if abs(prev_loss - loss.item()) < self.tolerance:
                    print(f"Converged after {_} iterations")
                    break   

                prev_loss = loss.item()


    def predict(self, X, threshold=0.5):
        y_predict_proba = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_predict_proba])


    def predict_proba(self, X):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model, self.b).detach().numpy() / b2c(self.b.detach().cpu().numpy())


    def get_c_log(self):
        if self.c_log is None:
            raise ValueError("c log is not available. Make sure to call fit() first.")
        return self.c_log


    def get_loss_c_log(self):
        if self.loss_c_log is None:
            raise ValueError("Loss c log is not available. Make sure to call fit() first.")
        return self.loss_c_log


    def get_c_hat(self):
        if self.b is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_b().")
        return b2c(self.b.detach().numpy())
    """
    s(x) = p(s=1|x)
    s(x) = probaility that a instance is labeled
    """
    def predict_label_proba(self, X):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model, self.b).detach().numpy()
        
    def predict_torch_label_proba(self,X_t):
        linear_model = self.update_linear_model(X_t)
        return self._activation(linear_model, self.b)