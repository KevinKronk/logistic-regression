import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

from cost import log_cost
from map_feature import map_feature
from predict import predict


# Logistic Regression using scipy.optimize.minimize and regularization


# Load Data

filename = 'microchip_tests.txt'
df = pd.read_csv(filename, header=None, names=['Test 1', 'Test 2', 'Accepted'])


# Plot Data

positive = df[df['Accepted'].isin([1])]
negative = df[df['Accepted'].isin([0])]

plt.scatter(positive['Test 1'], positive['Test 2'], s=15, c='b', marker='o', label='Accepted')
plt.scatter(negative['Test 1'], negative['Test 2'], s=15, c='r', marker='x', label='Not Accepted')

plt.legend()
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.show()


# Convert DataFrames to ndarrays

df.insert(0, 'Ones', 1)

cols = df.shape[1]
x = df.iloc[:, 0:cols-1]
y = df.iloc[:, cols-1:cols]

x = x.values
y = y.values

# Map Features (for microchip test)
x = map_feature(x[:, 1], x[:, 2], degree=6)

# Create Theta
theta = np.array([[0 for _ in range(x.shape[1])]])

# Set Hyperparameter
hyper_p = 0.01


# Minimize Function

options = {'maxiter': 400}
res = opt.minimize(log_cost, theta, (x, y, hyper_p), jac=True, method='TNC', options=options)

cost = res.fun
print(f"The current cost is: {cost}")
new_theta = res.x


# Determine Accuracy

accuracy = predict(new_theta, x, y)
print(f"Accuracy = {accuracy}%")
