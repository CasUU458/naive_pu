import numpy as np
import torch


def _loss (y_true, y_pred):

    eps = 1e-15  # to avoid log(0), numerical stability, small constant
    # ensure y_pred is in the range [eps, 1-eps]
    #clamp_min y_pred to avoid log(0)

    #positive contribution
    y_pred = y_pred.clamp(eps, 1. - eps)
    term1 = y_true* torch.log(y_pred)
    term2 = (1. - y_true) * torch.log(1. - y_pred)
    loss = -torch.mean(term1 + term2)
    return loss

def _sigmoid(z):
    return 1. / (1. + torch.exp(-z))


def _modified_pu_sigmoid(z, b):
    return 1. / (1. + torch.square(b) + torch.exp(-z))


# c is the label frequency
# b is the surrogate parameter for c, b = sqrt(1/c - 1)
def b2c(b):
    return 1 / (1 + b * b)


def c2b(c):
    return np.sqrt(1 / c - 1)
