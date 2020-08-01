# Gaussian discriminant analysis
import numpy as np
import utils
import matplotlib.pyplot as plt
# class
K = 2
N = 50
N0 = 25
N1 = 25
x_train, y_train = utils.create_toy_data()
print(x_train)
x_train_f = utils.feature(x=x_train, ftype='poly', degree=2)
figure = plt.figure()

x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
x_test = utils.feature(x=x_test, ftype='poly', degree=2)

x_train_f_0 = x_train_f[0:N0]
x_train_f_1 = x_train_f[N1:N]
u0 = np.mean(x_train_f_0, axis=0)
u1 = np.mean(x_train_f_1, axis=0)

x_mean_0 = x_train_f_0 - u0
x_mean_1 = x_train_f_1 - u1

S0 = (x_mean_0.T @ x_mean_0) / N0
S1 = (x_mean_1.T @ x_mean_1) / N1
S = S0 * N0 / N + S1 * N1 / N

S_inv = np.linalg.inv(S)
w = S_inv @ (u0 - u1)
w0 = -u0.T @ S_inv @ u0 + u1.T @ S_inv @ u1 + 2 * np.log(N0 / N1)
w0 /= 2
y_pre = utils.sigmoid(x_test @ w + w0)
y_test = (y_pre > 0.5).astype(np.int)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_test.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

