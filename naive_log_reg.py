import torch
import numpy as np


# c is the label frequency # b is the surrogate parameter for c, b = sqrt(1/c - 1)
def b2c(b):
    return 1 / (1 + b * b)


def c2b(c):
    return np.sqrt(1 / c - 1)


# loss function binary cross entropy or log likelihood?
def _loss(y_true, y_pred):
    eps = 1e-15  # to avoid log(0), numerical stability, small constant
    # ensure y_pred is in the range [eps, 1-eps]
    # clamp_min y_pred to avoid log(0)

    # positive contribution
    y_pred = y_pred.clamp(eps, 1. - eps)
    term1 = y_true * torch.log(y_pred)
    term2 = (1. - y_true) * torch.log(1. - y_pred)
    loss = -torch.mean(term1 + term2)
    return loss


# modified sigmoid function, as suggested in the paper
def _modified_sigmoid(z, b):
    out = 1. / (1. + torch.square(b) + torch.exp(-z))

    return out


class NaiveLogReg:
    def __init__(self, learning_rate=0.001, epochs=1000, device='cpu', tolerance=1e-6, c_estimate=None,
                 learning_rate_c=None, random_state=42):
        self.optimizer = None
        self.optimizer_b = None
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.device = device
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.tolerance = tolerance
        self.loss_log = None

        # !!   NAIVE MODIFICATIONS !!
        if c_estimate is not None:
            self.c_estimate = c_estimate
        else:
            rng = np.random.default_rng(seed=random_state)
            self.c_estimate = rng.uniform(0.1, 0.9)  # Randomly initialize c_estimate between 0.1 and 0.9

        if learning_rate_c is not None:
            self.learning_rate_c = learning_rate_c
        else:
            # Set learning rate for c based on epochs, suggest by paper
            self.learning_rate_c = 1 / epochs

        self.b = None
        self.b_init = c2b(self.c_estimate)

        self.loss_c_log = None
        self.c_log = None

    def fit(self, X, y):
        num_samples, n_features = X.shape
        # self.weights = torch.zeros((num_features,1), dtype=torch.float32,requires_grad=True)
        # self.bias = torch.zeros(1, dtype=torch.float32,requires_grad=True)

        # y_t = torch.as_tensor(y, dtype=torch.float32)

        self.weights = torch.zeros(n_features, device=self.device, requires_grad=True)
        self.bias = torch.zeros(1, device=self.device, requires_grad=True)
        # b parameter as surrogate for c
        self.b = torch.tensor(self.b_init, device=self.device, requires_grad=True)

        # Convert labels to 1-D
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)
        self.optimizer_b = torch.optim.Adam([self.b], lr=self.learning_rate_c)

        prev_loss = float('inf')

        # Track the loss progression for analysis, loss for s(x), y(x) and the value of c itself
        self.loss_log = np.zeros(self.epochs)
        self.loss_c_log = np.zeros(self.epochs)
        self.c_log = np.zeros(self.epochs)

        for _ in range(self.epochs):
            linear_model = X_t @ self.weights + self.bias
            y_predicted = _modified_sigmoid(linear_model, self.b)

            # loss = torch.nn.functional.binary_cross_entropy(y_predicted, y_t,reduction='mean')
            loss = _loss(y_t, y_predicted)
            # reset the gradients to zero
            self.optimizer.zero_grad()
            # calculate gradients
            loss.backward()
            # update the weights and bias
            self.optimizer.step()

            # Log the loss progression
            self.loss_log[_] = loss.item()

            # NAIVE B OPTIMIZATION !!
            linear_model = X_t @ self.weights + self.bias
            y_predicted = _modified_sigmoid(linear_model, self.b)

            loss_b = _loss(y_t, y_predicted)
            self.optimizer_b.zero_grad()
            loss_b.backward()
            self.optimizer_b.step()

            # Log the loss progression for b
            self.loss_c_log[_] = loss_b.item()
            self.c_log[_] = b2c(self.b.detach().numpy())

            if _ % 1000 == 0:
                print(
                    f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()} Loss b: {loss_b.item()}, c: {b2c(self.b.detach().cpu().numpy())}")
            # check for convergence
            if abs(prev_loss - loss.item()) < self.tolerance:
                print(f"Converged after {_} iterations")
                break

        return self

    def predict(self, X, th=0.5):
        X = torch.tensor(X, dtype=torch.float32)

        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")

        # y(x) = P(s=1 | x) / c
        linear_model = X @ self.weights + self.bias
        y_predicted = _modified_sigmoid(linear_model, self.b).detach().numpy() / b2c(self.b.detach().cpu().numpy())
        return np.array([1 if i > th else 0 for i in y_predicted])

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)

        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before predict().")

        linear_model = X @ self.weights + self.bias
        y_predicted = _modified_sigmoid(linear_model, self.b).detach().numpy() / b2c(self.b.detach().cpu().numpy())
        return y_predicted

    def get_loss_log(self):
        if self.loss_log is None:
            raise ValueError("Loss log is not available. Make sure to call fit() first.")
        return self.loss_log

    def get_weights(self):
        if self.weights is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_weights().")
        return self.weights.detach().cpu().numpy()

    def get_bias(self):
        if self.bias is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_bias().")
        return self.bias.detach().cpu().numpy()

    def get_c_log(self):
        if self.c_log is None:
            raise ValueError("c log is not available. Make sure to call fit() first.")
        return self.c_log

    def get_loss_c_log(self):
        if self.loss_c_log is None:
            raise ValueError("Loss c log is not available. Make sure to call fit() first.")
        return self.loss_c_log

    def c_hat(self):
        if self.b is None:
            raise ValueError("Model has not been trained yet. Call fit() before get_b().")
        return b2c(self.b.detach().numpy())
