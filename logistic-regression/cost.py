import numpy as np
from logistic_regression import sigmoid


def log_cost(x, y, theta):

    size = y.shape[0]
    # im assuming the matrices have been converted to ndarray

    h = sigmoid(x @ theta.T)
    cost = -((1 / size) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))
    return cost
