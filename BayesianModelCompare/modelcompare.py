import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
import utils
alpha = 1
beta = 0.5

x_train = np.random.rand(1000, 2) * 5
mu_real = np.random.rand(10, 2)*5
y_train = utils.mixgaussian(x=x_train, mu=mu_real)

evidence = []
for i in range(10):
    # get design matrix
    x_bf_train = utils.expand_gaussian_bf(x=x_train, bfnumber=i+1)
    evidence.append(utils.log_margin_likelihood(Phi=x_bf_train, t=y_train, alpha=alpha, beta=beta))

plt.plot(evidence)
plt.show()



