# using least square classification to classify K classes
# @author ljh
import numpy as np
import utils
import matplotlib.pyplot as plt
# class number
K = 2

x_train, y_train = utils.create_toy_data()
x_train_f = utils.feature(x=x_train, ftype='poly', degree=2)
figure = plt.figure()

x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
x_test = utils.feature(x=x_test, ftype='poly', degree=2)


# encode y_train into one-of-K encoding
y_train_one_hot = np.eye(N=K)[y_train]

# compute W of linear classification function
# formula: W = X_pinv * T
# X_pinv: the Pseudo-inverse of matrix X (whose n-th row is x_n^T), in a word it equals to pseudo-inverse of x_train_f
# page 185 of PRML
W = np.linalg.pinv(x_train_f) @ y_train_one_hot

y_test = x_test @ W
y_test = np.argmax(y_test, axis=1)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_test.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()