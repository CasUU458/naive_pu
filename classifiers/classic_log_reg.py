import torch 
import numpy as np

from classifiers.helpers import _loss, _sigmoid,penalty
from classifiers.base_log_reg import BaseLogReg
from config import CONFIG


class ClassicLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=1000, tolerance=1e-6,penalty=None):

        super().__init__(learning_rate, epochs, tolerance, _sigmoid,penalty)


    def fit(self, X, y):
        num_samples, n_features = X.shape

        self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
        self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

        X_t = torch.as_tensor(X, dtype=torch.float32,device=CONFIG.TORCH_DEVICE)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)

        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
        
        prev_loss = float('inf')

        self.loss_log = np.zeros(self.epochs)

        for _ in range(self.epochs):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = self._activation(linear_model)
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()}")

            loss = _loss(y_t, y_predicted)
            loss = penalty(self.penalty, loss, self.weights)

            self.optimizer.zero_grad() # reset grads
            loss.backward() # calculate grads
            self.optimizer.step() # update weights and bias

            self.loss_log[_] = loss.item() # log loss

            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break   

        return self


    def predict(self, X, threshold=0.5):
        y_predict_proba = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_predict_proba])


    def predict_proba(self, X):
        linear_model = self.update_linear_model(X)
        return self._activation(linear_model).detach().numpy()
