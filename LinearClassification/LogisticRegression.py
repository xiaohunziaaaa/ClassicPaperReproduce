# using logistic regression to do classification
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


# Using Iterative Reweighted Least Square (IRLS) to get solution
w = np.zeros(np.size(x_train_f, 1))
max_iter = 100
for i in range(max_iter):
    w_prev = np.copy(w)
    y_prev = utils.sigmoid(x_train_f @ w)
    grad_E_W = x_train_f.T @ (y_prev - y_train)
    # without optimizaiton
    R = np.diag(y_prev * (1-y_prev))
    Hessian = (x_train_f.T @ R) @ x_train_f
    # with optimization
    # Hessian = (x_train_f.T * y_prev * (1-y_prev)) @ x_train_f
    try:
        w -= np.linalg.solve(Hessian, grad_E_W)
    except np.linalg.LinAlgError:
        print('Error! Iteration {}.'.format(i))
        break
    if np.allclose(w, w_prev):
        break

# testing
y_test = (utils.sigmoid(x_test @ w) > 0.5).astype(np.int)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_test.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
