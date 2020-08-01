'''
Using iteration method to evaluate alpha and beta to maximize evidence
Modified from https://zjost.github.io/bayesian-linear-regression/
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
import utils

x_train = np.random.rand(1000, 2) * 5
mu_real = np.random.rand(10, 2)*5
y_train = utils.mixgaussian(x=x_train, mu=mu_real)

x_bf_train = utils.expand_gaussian_bf(x=x_train, bfnumber=10)
alpha, beta, m_N, S_N = utils.fit(Phi=x_bf_train, t=y_train, max_iter=200, rtol=1e-4)
print(alpha, beta)

x_test = np.random.rand(1000, 2) * 5
x_bf_test = utils.expand_gaussian_bf(x=x_test, bfnumber=10)
y_test = utils.mixgaussian(x=x_test, mu=mu_real)

y_pre, y_var_pre = utils.prediction(phi_test=x_bf_test, m_N=m_N, S_N=S_N, beta=beta)
print(y_pre.shape)

utils.plot_3D(x_test, y_test)
utils.plot_3D(x_test, y_pre)