import numpy as np


def map_feature(x1, x2, degree=6):
    """
        Maps the two input features to quadratic features used in logistic regression.

        Parameters
        ----------
        x1 : array_like
            Shape (m, 1), containing one feature.

        x2 : array_like
            Shape (m, 1), containing a second feature of the same size.

        degree : int, optional
            The polynomial degree.

        Returns
        -------
        : array_like
            A matrix of m rows, and columns depending on degree of polynomial.
    """

    if x1.ndim > 0:
        out = [np.ones(x1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((x1 ** (i - j)) * (x2 ** j))

    if x1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)
