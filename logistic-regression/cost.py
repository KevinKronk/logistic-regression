import numpy as np

from gradient import gradient
from sigmoid import sigmoid


def log_cost(theta, x, y, hyper_p):
    """
        Logistic regression cost function with regularization.

        Parameters
        ----------
        theta : array_like
            Shape (1, n+1). Parameter values for function.

        x : array_like
            Shape (m, n+1). Features in model.

        y : array_like
            Shape (m, 1). Labels for each example.

        hyper_p : float
            Value of the hyperparameter for regularization.

        Returns
        -------
        cost : float
            Value of cost function at given parameters.

        grad : array_like
            Shape (1, n+1). The gradient for each parameter.
    """

    theta = np.reshape(theta, (1, x.shape[1]))
    size = y.shape[0]

    h = sigmoid(x @ theta.T)
    reg = (hyper_p / 2 * size) * np.sum(theta[:, 1:theta.shape[1]] ** 2)

    cost = -((1 / size) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))) + reg
    grad = gradient(theta, x, y, hyper_p)
    return cost, grad
