from abc import abstractmethod, ABC

import numpy as np
import torch

from classifiers.helpers import b2c


class BaseLogReg(ABC):
    def __init__(self, learning_rate, epochs, tolerance, activation,penalty):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance = tolerance
        self.penalty = penalty
        
        self._activation = activation

        self.optimizer = None
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

        self.loss_log = None


    @abstractmethod
    def fit(self, X, y):
        # require all subclasses to override this method.
        pass


    @abstractmethod
    def predict(self, X, threshold=0.5):
        # require all subclasses to override this method.
        pass


    @abstractmethod
    def predict_proba(self, X):
        # require all subclasses to override this method.
        pass


    def update_linear_model(self, X):
        X = torch.tensor(X, dtype=torch.float32)

        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")

        return X @ self.weights + self.bias


    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_weights().")
        return self.weights.detach().cpu().numpy()


    def get_bias(self):
        if self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_bias().")
        return self.bias.detach().cpu().numpy()


    def get_loss_log(self):
        if self.loss_log is None:
            raise ValueError("Loss log is not available. Make sure to call fit() first.")
        return self.loss_log