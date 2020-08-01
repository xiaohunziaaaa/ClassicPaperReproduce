import numpy as np
import itertools
import functools
import math

def create_toy_data(add_outliers=False, add_class=False):
    '''
    :param add_outliers:
    :param add_class:
    :return:
    : stolen from https://nbviewer.jupyter.org/github/ctgk/PRML/blob/master/notebooks/ch04_Linear_Models_for_Classfication.ipynb
    '''
    x0 = np.random.normal(size=50).reshape(-1, 2) - 1
    x1 = np.random.normal(size=50).reshape(-1, 2) + 1.
    if add_outliers:
        x_1 = np.random.normal(size=10).reshape(-1, 2) + np.array([5., 10.])
        return np.concatenate([x0, x1, x_1]), np.concatenate([np.zeros(25), np.ones(30)]).astype(np.int)
    if add_class:
        x2 = np.random.normal(size=50).reshape(-1, 2) + 3.
        return np.concatenate([x0, x1, x2]), np.concatenate([np.zeros(25), np.ones(25), 2 + np.zeros(25)]).astype(np.int)
    return np.concatenate([x0, x1]), np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int)

def feature(x, ftype ='poly', degree=2, w0=False):
    '''
    :param x: array needed to be extend
    :param ftype: feature type, polynomial supported only, this parameters will be used in future
    :param degree: the degree of polynoial
    :return:
    '''
    if x.ndim == 1:
        x = x[:, None]
    x_t = x.transpose()
    # this code will cause (x-u)'(x-u) to be singular
    features = [np.ones(len(x))]

    for dg in range(1, degree + 1):
        # extend x to corresponding degree
        for items in itertools.combinations_with_replacement(x_t, dg):
            # do multiply
            features.append(functools.reduce(lambda x, y: x * y, items))
    result = np.asarray(features).transpose()
    if w0:
        return result
    else:
        return result[:, 1:-1]


def PCA(x, dim = 1):
    assert dim < len(x)
    x_central = x - np.mean(a=x, axis=0)
    cov = np.cov(m=x_central, rowvar=False)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    eigen_vec = eigen_vec[:, 0:dim]

    return eigen_val, eigen_vec, x @ eigen_vec

def sigmoid(x):
  return np.tanh(x * 0.5) * 0.5 + 0.5