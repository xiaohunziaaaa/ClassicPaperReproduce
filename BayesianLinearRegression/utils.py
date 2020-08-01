import pandas as pd
import numpy as np

'''
@df: input data frame in pandas df format
return: training and testing data set in numpy array format
'''
def format_data(df):
    labels = df['G3']
    df = df.drop(columns=['school', 'G1', 'G2'])
    df = pd.get_dummies(df)
    most_correlated = df.corr().abs()['G3'].sort_values(ascending=False)
    most_correlated = most_correlated[:8]
    df = df.loc[:, most_correlated.index]
    # higher yes and higher no mean same thing
    df = df.drop(columns='higher_no')
    ndf = df.values
    st_train = ndf[0:int(len(ndf) * 0.7)]
    st_test = ndf[int(len(ndf) * 0.7):]
    st_train_y = st_train[:, 0:1]
    st_train_x = st_train[:, 1:]
    st_test_y = st_test[:, 0:1]
    st_test_x = st_test[:, 1:]
    return st_train_x, st_train_y, st_test_x, st_test_y

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
    # unnecessary computation of cov(y_i, y_j), need to be optimized
    y_var = 1/beta + np.matmul(np.matmul(phi_test, S_N), phi_test.T)
    return y, y_var

