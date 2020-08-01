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

# Using perceptron to do classification
max_iter = 1000
lr = 0.1
w = np.zeros(np.size(x_train_f, 1))
# change y label {0, 1} to {-1, 1}
y_train[0:25] = y_train[0:25] - 1
for i in range(max_iter):
    x_error = x_train_f[np.sign(x_train_f @ w) != y_train]
    y_error = y_train[np.sign(x_train_f @ w) != y_train]
    idx = np.random.choice(len(x_error))
    w += lr * x_error[idx] * y_error[idx]
    if (x_train_f @ w * y_train > 0).all():
        break

# testing
print(x_error @ w)
y_test = ((x_test @ w) > 0).astype(np.int)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
plt.contourf(x1_test, x2_test, y_test.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()