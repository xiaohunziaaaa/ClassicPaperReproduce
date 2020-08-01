import numpy as np

def init(shape, method='normal'):
    if method == 'normal':
        return np.random.normal(size=shape)
    if method == 'zeros':
        return np.zeros(shape=shape)

    return np.random.normal(size=shape)