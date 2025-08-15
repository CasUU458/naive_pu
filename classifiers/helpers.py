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

def penalty(type,loss,weights):
    if str(type).lower() == "l1":
        return loss + l1_penalty(weights)
    elif str(type).lower() == "l2":
        return loss + l2_penalty(weights)
    return loss    

def l1_penalty(weights, C=1):
    
    """
    Compute the L1 regularization penalty for given model weights.
    L1 penalty: mean of absolute weight values

    Drives weights to zero!

    Parameters:
    weights: Model weight tensor (bias should be excluded).

    C (float): Inverse regularization strength, Larger C => weaker regularization. Default = 1.
    """
    
    lamda = 1 / C 
    return lamda * torch.mean(torch.abs(weights))


def l2_penalty(weights, C=1):
    """
    Compute the L2 regularization penalty for given model weights.
    L2 penalty: mean of squared weight values
    
    Parameters:
    weights: Model weight tensor (bias should be excluded).
    C (float) : Inverse regularization strength, Larger C => weaker regularization. Default = 1.

    """
    lamda = 1 / C  
    
    return lamda * torch.mean(weights ** 2)


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
