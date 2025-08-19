import torch
import numpy as np

from classifiers.helpers import _loss,penalty
from classifiers.base_log_reg import BaseLogReg
from classifiers.helpers import c2b, _modified_pu_sigmoid, b2c
from config import CONFIG
import time
import logging


class NaiveLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=1000, tolerance=1e-6, c_estimate=None, learning_rate_c=None,penalty=None,solver='adam'):

        super().__init__(learning_rate, epochs, tolerance, _modified_pu_sigmoid,penalty,solver)

        self.optimizer_b = None

        if c_estimate is not None:
            self.c_estimate = c_estimate
        else:
            rng = np.random.default_rng(seed=CONFIG.SEED)
            self.c_estimate = rng.uniform(0.1, 0.9)  # Randomly initialize c_estimate between 0.1 and 0.9

        if learning_rate_c is not None:
            self.learning_rate_c = learning_rate_c
        else:
            # Set learning rate for c based on epochs, as suggest in paper
            self.learning_rate_c = 1 / epochs

        self.b = None # b parameter as surrogate for c
        self.b_init = c2b(self.c_estimate)

        self.c_log = None
        self.loss_c_log = None
    


    def fit(self, X, y):
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
        logging.info(f"NaiveLogReg completed in {elapsed:.4f} seconds")
        return result

    def fit_adam(self, X, y):
        num_samples, n_features = X.shape

        self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
        self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

        X_t = torch.as_tensor(X, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)

        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)

        self.b = torch.tensor(self.b_init, device=CONFIG.TORCH_DEVICE, requires_grad=True)
        self.optimizer_b = torch.optim.Adam([self.b], lr=self.learning_rate_c)

        prev_loss = float('inf')
        self.loss_log, self.loss_c_log, self.c_log = np.zeros(self.epochs), np.zeros(self.epochs), np.zeros(self.epochs)

        for _ in range(self.epochs):
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
            self.c_log[_] = b2c(self.b.detach().numpy())

            if _ % 1000 == 0:
                print(
                    f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()} Loss b: {loss_b.item()}, c: {b2c(self.b.detach().cpu().numpy())}")

            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break

            prev_loss = loss.item()
        return self

    def fit_lbfgs(self, X, y):
            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

            X_t = torch.as_tensor(X, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)

            self.optimizer = torch.optim.LBFGS([self.weights, self.bias], lr=self.learning_rate)

            self.b = torch.tensor(self.b_init, device=CONFIG.TORCH_DEVICE, requires_grad=True)
            self.optimizer_b = torch.optim.LBFGS([self.b], lr=self.learning_rate_c)

            prev_loss = float('inf')
            self.loss_log, self.loss_c_log, self.c_log = np.zeros(self.epochs), np.zeros(self.epochs), np.zeros(self.epochs)

            
            for _ in range(self.epochs):
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
                self.c_log[_] = b2c(self.b.detach().cpu().numpy())

                if _ % 100 == 0:
                    print(f"Iteration {_}, Loss: {loss.item()}, Loss b: {loss_b.item()}, c: {b2c(self.b.detach().cpu().numpy())}")

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
    def predict_label_proba(self, X, threshold=0.5):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model, self.b).detach().numpy()
        
    def predict_torch_label_proba(self, X, threshold=0.5):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model, self.b)