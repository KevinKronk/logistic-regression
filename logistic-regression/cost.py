import numpy as np
from sigmoid import sigmoid


def log_cost(theta, x, y):
    theta = np.reshape(theta, (1, 3))
    size = y.shape[0]
    # im assuming the matrices have been converted to ndarray

    h = sigmoid(x @ theta.T)
    return -((1 / size) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
