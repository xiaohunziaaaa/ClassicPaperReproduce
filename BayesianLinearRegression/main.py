import utils
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
'''
using student data
# load data
dataset_path = '../Dataset/student/student-mat.csv'
stscore = pd.read_csv(filepath_or_buffer=dataset_path, delimiter=';')
# select most related filed and store their values into numpy array
# and then format it into standard training and testing data set
x_train, y_train, x_test, y_test = utils.format_data(df=stscore)

# model below assume alpha and peta is known, so using MAP to solve parameters. And then giving prediction.
# construct bayesian linear model, only consider direct linear relation to x_train, no basic function has been used

alpha = 1
beta = 0.5

# construct design matrix
Phi = x_train
t = y_train
m_N, S_N = utils.posterior(Phi=Phi, t=t, alpha=alpha, beta=beta, return_inverse=False)

phi_test = x_test[i]
phi_test = np.reshape(a=phi_test, newshape=[-1, 1])

y_predic, y_var_predic = utils.prediciton(phi_test=phi_test, m_N=m_N, S_N=S_N, beta=beta)
y_predic = np.asscalar(y_predic)
'''
# using simulated data
alpha = 1
beta = 0.5

x_train = np.random.rand(100, 2) * 5
real_W = np.array([[0.5], [2.1]])
y_train = np.matmul(x_train, real_W) + np.random.normal(loc=0, scale=0.5, size=[100, 1])

m_N, S_N = utils.posterior(Phi=x_train, t=y_train, alpha=alpha, beta=beta, return_inverse=False)

print('posterior distribution\'s mean ={}, variance = {}'.format(m_N, S_N))

x_test = np.random.rand(1000, 2) * 5
y_test = np.matmul(x_train, real_W) + np.random.normal(loc=0, scale=0.5, size=[100, 1])

y_pre, y_var_pre = utils.prediction(phi_test=x_test, m_N=m_N, S_N=S_N, beta=beta)
y_var_dia = np.diagonal(y_var_pre)

x_1 = x_test[:, 0]
x_2 = x_test[:, 1]
y = np.reshape(a=y_pre, newshape=[-1])


fig = plt.figure()
#创建一个三维坐标轴
ax = plt.axes(projection='3d')
#三角螺旋线
# 三维线的数据
ax.scatter3D(x_1, x_2, y, c=y, cmap='Greens')
plt.show()







