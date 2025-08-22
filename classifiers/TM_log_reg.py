import torch 
import numpy as np

from classifiers.helpers import _sigmoid, penalty, _loss
from classifiers.base_log_reg import BaseLogReg
from classifiers.naive_log_reg import NaiveLogReg
from classifiers.classic_log_reg import ClassicLogReg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import CONFIG
import time
import logging


class TwoModelLogReg(BaseLogReg):
    def __init__(self, learning_rate=0.001, epochs=300, tolerance=1e-6,penalty=None,solver='adam',alpha=0.5,epsilon=1e-8,validation=None):
        super().__init__(learning_rate, epochs, tolerance, _sigmoid,penalty,solver)
        # the model that predicts labels
        self.y = None
        self.e = None

        self.alpha = alpha # alpha is the quantile for the threshold
        self.epsilon = epsilon
        self.val_log = []
        self.VAL = [validation[0],validation[1],validation[2]] if validation is not None else None # validation set: X_val, y_val, s_val


    def fit(self, data_X, data_y, **kwargs):
        start = time.perf_counter()

        # first, we fit (and validate) the data with the naive classifier to initialize weights
        naive_clf = NaiveLogReg(epochs=300,penalty="l2",solver="adam")
        naive_clf.fit(data_X, data_y)

        # s and OR are required by y, we estimate them with the naive classifier
        s_naive = naive_clf.predict_torch_label_proba(data_X)
        e_naive = 0.5 * (s_naive + 1)
        OR = OddsRatio(e_naive, s_naive)

        # initialize the two models
        self.e = ClassicLogReg()
        self.y = self.Y()

        current_iteration = 0
        while current_iteration < self.epochs:

            self.y.fit(data_X, data_y, OR=OR) # solve βˆn
            y_x = self.y.predict_proba(data_X) # calculate yˆ(Xi)

            threshold = calculate_threshold(y_x, data_y, self.alpha)
            X_p, y_p = filter_samples(data_X, data_y, y_x, threshold)

            self.e.fit(X_p, y_p) # solve γˆn by using pseudo set
            e_x = self.e.predict_proba(data_X) # calculate eˆ(Xi)

            # s(x) = e(x)*y(x) = P(s= 1|x) = P(s = 1|,y = 1,x) * P(y=1|x)
            s = e_x * y_x
            OR = OddsRatio(e_x, s)

            if current_iteration % 10 == 0:
                print(f"Iteration {current_iteration}")

            if self.VAL is not None:
                self.validate(self.y,name="y(X)")
                self.validate(self.e,name="e(X)",label_freq=True)
                # self.val_log.append(("threshold",threshold,0,0,0))
                # self.val_log.append(("OR",OR,0,0,0))

            current_iteration += 1

        elapsed = time.perf_counter() - start
        logging.info(f"TwoModelLogReg completed in {elapsed:.4f} seconds")
        return self


    class Y(ClassicLogReg):
        """
        Adaptation of the classic logistic regression with weight adjustment, can be trained on positive and unlabeled data.
        Weights are based on the odds ratio 
        """
        def __init__(self, learning_rate=0.001, epochs=100, tolerance=0.001, penalty="l2", solver='lbfgs'):
            super().__init__(learning_rate, epochs, tolerance, penalty, solver)

        # Weight function unlabeled class
        def w0(self, y, OR):
            return (1. - y) * (1. - OR)

        # Weight function positive class
        def w1(self, y, OR):
            return y + ((1. - y) * OR)

        # Loss is adjusted based on class of samples
        def _weighted_loss(self, y_t, y_pred, OR):
            eps = 1e-15
            y_pred = y_pred.clamp(eps, 1. - eps)

            term1 = self.w1(y_t, OR) * torch.log(y_pred)
            term2 = self.w0(y_t, OR) * torch.log(1. - y_pred)
            return -torch.mean(term1 + term2)

        # Adaptation of the classic logistic regression fit function with adam solver
        # Loss is now with weight adjustment
        def fit(self, X, y, **kwargs):
            OR = kwargs.get("OR", None)

            num_samples, n_features = X.shape

            self.weights = torch.zeros(n_features, device=CONFIG.TORCH_DEVICE, requires_grad=True)
            self.bias = torch.zeros(1, device=CONFIG.TORCH_DEVICE, requires_grad=True)

            X_t = torch.as_tensor(X, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=CONFIG.TORCH_DEVICE)

            self.optimizer = torch.optim.Adam([self.weights, self.bias], lr=self.learning_rate)

            prev_loss = float('inf')

            self.loss_log = np.zeros(self.epochs)

            for _ in range(self.epochs):
                linear_model = X_t @ self.weights + self.bias
                y_predicted = self._activation(linear_model)
                # if _ % 100 == 0:
                #     print(f"Iteration {_}, Loss: {_loss(y_t, y_predicted).item()}")

                loss = self._weighted_loss(y_t, y_predicted, OR)
                loss = penalty(self.penalty, loss, self.weights)

                self.optimizer.zero_grad()  # reset grads
                loss.backward()  # calculate grads
                self.optimizer.step()  # update weights and bias

                self.loss_log[_] = loss.item()  # log loss

                if abs(prev_loss - loss.item()) < self.tolerance:
                    # print(f"Converged after {_} iterations")
                    break

                prev_loss = loss.item()

            return self


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


    def predict(self, X, threshold=0.5):
        return self.y.predict(X, threshold)


    def predict_proba(self, X):
        return self.y.predict_proba(X)


    def get_weights(self):
        return self.y.get_weights()


    def get_e_weights(self):
        return self.e.get_weights()


    def _get_y_clf(self):
        return self.y


    def _get_e_clf(self):
        return self.e


    def get_validation_logs(self):
        return self.val_log


    def validate2log(self):
        for i,log in enumerate(self.val_log):
            logging.info(f"Validation log {i}: {log}")


def OddsRatio(e, s):
    """
    Estimates the odds ratio for a given sample.
    Is the ratio between the odds of sample being unlabeled among the positives versus the odds of a sample being
    unlabeled among the the complete set of both positives and negatives.
    """
    OR = ((1 - e) / e) / ((1 - s) / s)
    if isinstance(OR, torch.Tensor):
        return OR.detach().numpy()
    return OR # must be a numpy array to avoid loss function calculating grads on the OR tensor.


def calculate_threshold(y_x, y, alpha):
    idxs = []
    for idx, y in enumerate(y):
        if y == 1:
            idxs.append(idx)

    return np.quantile(y_x[idxs], alpha)


def filter_samples(X, y, y_x, threshold):
    """
    Define the pseudo-label set based on the current threshold.
    samples with predicted probabilities above the threshold are considered positive.
    """
    idxs = []
    for idx, (loop_y, loop_y_x) in enumerate(zip(y, y_x)):
        if loop_y == 1 or loop_y_x > threshold:
            idxs.append(idx)

    return X[idxs], y[idxs]
