# implementation of Linear Discriminant Analysis(LDA), aka Fisher Discriminant Analysis
import utils
import numpy as np
import matplotlib.pyplot as plt
x_train, y_train = utils.create_toy_data()
x_train_f = utils.feature(x=x_train, ftype='poly', degree=2)

# solve Sw
x_a = x_train_f[0:25]
x_b = x_train_f[25:50]

# do PCA to avoid singular S_W(when samples less than features or else), this technique is known as Fisherface
eigen_val, eigen_vec, x_a = utils.PCA(x_a, dim=5)
eigen_val, eigen_vec, x_b = utils.PCA(x_b, dim=5)

m_a = np.mean(a=x_a, axis=0)
m_b = np.mean(a=x_b, axis=0)
x_a_central = x_a - m_a
x_b_central = x_b - m_b
S_W = (x_a_central.T @ x_a_central + x_b_central.T @ x_b_central)
w = np.linalg.solve(S_W, (m_a - m_b))
w /= np.linalg.norm(w).clip(min=1e-10)

# testing and visualization
x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
x_test = np.array([x1_test, x2_test]).reshape(2, -1).T
x_test = utils.feature(x=x_test, ftype='poly', degree=2)

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)

x_test = x_test @ eigen_vec
y_test = x_test @ w
# solve threshold
# directly choose the mean value
# threshold = np.mean(y_test)
# using method from https://github.com/ctgk/PRML/blob/master/prml/linear/fishers_linear_discriminant.py
# to determine threshold
y_a_pre = x_a @ w
y_b_pre = x_b @ w
y_a_mean = np.mean(y_a_pre)
y_b_mean = np.mean(y_b_pre)
y_a_var = np.var(y_a_pre)
y_b_var = np.var(y_b_pre)

root = np.roots([
    y_b_var - y_a_var,
    2 * (y_a_var * y_b_mean - y_b_var * y_a_mean),
    y_b_var * y_a_mean ** 2 - y_a_var * y_b_mean ** 2
    - y_b_var * y_a_var * np.log(y_b_var / y_a_var)
])

if y_a_mean < root[0] < y_b_mean or y_b_mean < root[0] < y_a_mean:
        hreshold = root[0]
else:
        threshold = root[1]



y_test = (y_test > threshold).astype(np.int)
plt.contourf(x1_test, x2_test, y_test.reshape(100, 100), alpha=0.2, levels=np.linspace(0, 1, 3))
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


