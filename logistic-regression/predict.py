from sigmoid import sigmoid


def predict(theta, x, y):
    """
        Gives the accuracy of the model based on the theta parameters.

        Parameters
        ----------
        theta : array_like
            Shape (1, n+1). Parameter values for function.

        x : array_like
            Shape (m, n+1). Features in model.

        y : array_like
            Shape (m, 1). Labels for each example.

        Returns
        -------
        accuracy : int
            Percentage of correct model predictions.
    """

    correct = 0

    probability = sigmoid(x @ theta.T)
    predictions = [1 if x >= 0.5 else 0 for x in probability]

    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct += 1
    accuracy = int((correct / len(y)) * 100)
    return accuracy
