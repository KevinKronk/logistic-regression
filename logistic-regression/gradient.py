import numpy as np
from sigmoid import sigmoid


def gradient(theta, x, y, hyper_p):
    # Create temporary array for updating theta
    theta = np.reshape(theta, (1, x.shape[1]))
    size = y.shape[0]
    parameters = x.shape[1]
    grad = np.zeros(parameters)

    error = (1 / size) * (sigmoid(x @ theta.T) - y)

    for parameter in range(parameters):
        delta = error * x[:, [parameter]]

        if parameter == 0:
            grad[parameter] = delta.sum()
        else:
            grad[parameter] = delta.sum() + ((hyper_p / size) * theta[:, parameter])

    return grad

