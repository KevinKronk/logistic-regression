import numpy as np

from sigmoid import sigmoid


def gradient(theta, x, y, hyper_p):
    """
        Gradient with regularization for parameters in logistic regression.

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
        grad : array_like
            Shape (1, n+1). The gradient for each parameter.
    """

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
