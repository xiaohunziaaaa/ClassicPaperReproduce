import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d

def gaussian_basis_function(x, mu, sigma=0.1):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

def posterior(Phi, t, alpha, beta, return_inverse=False):
    '''
    :param Phi: design matrix
    :param t: target values
    :param alpha: precision parameter of prior distribution of W
    :param beta: precision parameter of predictive distribution(Conditional Gaussian Distribution)
    :param return_inverse: flag to indicate whether return precision(True) or variance(True)
    :return: Gaussian distribution parameters of posterior: mean and precision/variance
    '''
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * np.matmul(Phi.T, Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)
    if return_inverse:
        return m_N, S_N_inv
    else:
        return m_N, S_N

def prediction(phi_test, m_N, S_N, beta):
    '''
    :param phi_test: test input for regression task, basic function(lowercase) but not design matrix(uppercase)
    :param m_N: mean of posterior distribution
    :param S_N: variance of posterior distribution
    :param beta: precision of target distribution
    :return: mean and variance of predictive prediction
    '''
    print(phi_test.shape)
    y = np.matmul(phi_test, m_N)
    y_var = 1/beta + np.matmul(np.matmul(phi_test,S_N), phi_test.T)
    return y, y_var

def log_margin_likelihood(Phi, t, alpha, beta):
    '''
    :param Phi: design matrix
    :param t: target value
    :param alpha: precision of prior distribution
    :param beta: precision of predictive distribution
    :return: evidence
    '''

    # N = training samples, M = number of basis functions
    N, M = Phi.shape
    m_N, A = posterior(Phi=Phi, t=t, alpha=alpha, beta=beta, return_inverse=True)
    E_m_N = beta * np.sum((t - Phi.dot(m_N))**2) + alpha * np.sum(m_N ** 2)
    evidence = M * np.log(alpha) + N * np.log(beta) - E_m_N - np.log(np.linalg.det(A)) - N * np.log(2 * np.pi)
    return 0.5 * evidence

def mixgaussian(x, mu):
    y = []
    for i in range(len(mu)):
        mu_ = mu[i]
        rv = multivariate_normal(mean=mu_)
        y.append(rv.pdf(x))
    y = np.asarray(y).T
    y = np.sum(y, axis=1) + np.random.normal(loc=0, scale=0.5)
    return y

def expand_gaussian_bf(x, bfnumber):
    '''
    :param x: input
    :param bfnumber: number of basic (Gaussian) functions
    :return: design matrix
    '''
    mu = np.asarray([[0.24078802, 0.64845271],
                     [0.06130151, 0.92280903],
                     [0.08580807, 0.73550057],
                     [0.87034497, 0.97446972],
                     [0.54913108, 0.56968786],
                     [0.40955293, 0.49923087],
                     [0.74456808, 0.69659865],
                     [0.01378522, 0.55121218],
                     [0.96306163, 0.33291994],
                     [0.11169948, 0.39355584]])
    mu = mu * 5
    mu = mu[0:bfnumber]
    x_bf = []
    for i in range(len(mu)):
        mu_ = mu[i]
        rv = multivariate_normal(mean=mu_)
        x_bf.append(rv.pdf(x))

    x_bf = np.asarray(x_bf).T
    return x_bf

def plot_3D(x, y):
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], y, c=y, cmap='Blues')
    plt.show()

def fit(Phi, t, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5):
    '''
    :param Phi: design matrix
    :param t: target value
    :param alpha_0: initial alpha
    :param beta_0: initial beta
    :param max_iter: Maximum number of iterations.
    :param rtol: Convergence criterion.
    :return: alpha and beta that maximize evidence as well as corresponding posterior mean and posterior covariance
    '''
    N, M = Phi.shape
    # compute eigenvalues of beta*Phi'*Phi
    eigenvalues_0 = np.linalg.eigvalsh(beta_0 * np.matmul(Phi.T, Phi))
    beta = beta_0
    alpha = alpha_0
    for i in range(max_iter):
        beta_pre = beta
        alpha_pre = alpha
        eigenvalues = eigenvalues_0
        m_N, S_N = posterior(Phi=Phi, t=t, alpha=alpha, beta=beta)
        # compute gamma
        gamma = np.sum(eigenvalues/(eigenvalues + alpha))
        alpha = gamma / np.sum(m_N ** 2)
        beta_inv = 1 / (N - gamma) * np.sum((t - Phi.dot(m_N)) ** 2)
        beta = 1 / beta_inv

        if np.isclose(alpha_pre, alpha, rtol=rtol) and np.isclose(beta_pre, beta, rtol=rtol):
            print('Convergence after {} iterations.'.format(i + 1))
            return alpha, beta, m_N, S_N
    print('Stopped after {} iterations.'.format(max_iter))
    return alpha, beta, m_N, S_N

