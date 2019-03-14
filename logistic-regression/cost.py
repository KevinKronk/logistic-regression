import numpy as np
from sigmoid import sigmoid
from gradient import gradient


def log_cost(theta, x, y, hyper_p):
    theta = np.reshape(theta, (1, x.shape[1]))
    size = y.shape[0]
    # im assuming the matrices have been converted to ndarray

    h = sigmoid(x @ theta.T)
    reg = (hyper_p / 2 * size) * np.sum(theta[:, 1:theta.shape[1]] ** 2)

    cost = -((1 / size) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))) + reg
    grad = gradient(theta, x, y, hyper_p)
    return cost, grad
