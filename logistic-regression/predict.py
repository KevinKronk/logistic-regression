from sigmoid import sigmoid


def predict(theta, x, y):
    correct = 0

    probability = sigmoid(x @ theta.T)
    predictions = [1 if x >= 0.5 else 0 for x in probability]

    for i in range(len(y)):
        if predictions[i] == y[i]:
            correct += 1
    accuracy = correct / len(y)
    return accuracy
