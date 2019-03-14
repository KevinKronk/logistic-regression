import numpy as np


def sigmoid(z):
    """
        Applies logistic function to value.

        Parameter
        ---------
        z : int, array_like
            Single integer or array of values.

        Returns
        -------
        : int, array_like
            Single integer or array of values with logistic function applied.
    """

    return 1 / (1 + np.exp(-z))
