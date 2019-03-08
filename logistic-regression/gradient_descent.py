import numpy as np
from cost import log_cost
from sigmoid import sigmoid


def gradient_descent(x, y, theta, alpha, iterations):
    # Create temporary array for updating theta
    size = y.shape[0]
    parameters = x.shape[1]
    temp = np.zeros((1, parameters))
    cost_history = np.zeros(iterations)

    for iteration in range(iterations):

        error = (1 / size) * (sigmoid(x @ theta.T) - y)

        for parameter in range(parameters):
            delta = error * x[:, [parameter]]
            temp[0, parameter] = theta[0, parameter] - (alpha * delta.sum())

        theta = temp
        cost_history[iteration] = log_cost(x, y, theta)

    return theta, cost_history
