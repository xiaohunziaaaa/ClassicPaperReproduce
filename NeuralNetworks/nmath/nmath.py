import numpy as np


def softmax(a):
    """Compute softmax values for each sets of scores in x."""

    assert a.shape[1]
    e_a = np.exp(a)
    scale = 1/np.sum(e_a, axis=1)
    for i in range(a.shape[1]):
        e_a[:, i] *= scale
    return e_a


def dif_activation(a, act_type='sigmoid'):
    if act_type == 'sigmoid':
        return sigmoid(a) * (1 - sigmoid(a))
    if act_type == 'relu':
        return a
    if act_type == 'tanh':
        return a
    return a


def activation(a, act_type='sigmoid'):
    if act_type == 'sigmoid':
        return sigmoid(a)
    if act_type == 'relu':
        return relu(a)
    if act_type == 'tanh':
        return tanh(a)
    return a

def sigmoid(a):
    result = 1/(1 + np.exp(-a))
    return result

def relu(a):
    return a

def tanh(a):
    return a
