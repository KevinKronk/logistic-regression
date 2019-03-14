import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cost import log_cost
import scipy.optimize as opt
from predict import predict
from map_feature import map_feature

# Load Data

filename = 'ex2data2.txt'
df = pd.read_csv(filename, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


# Plot Data
'''
positive = df[df['Admitted'].isin([1])]
negative = df[df['Admitted'].isin([0])]

plt.scatter(positive['Exam 1'], positive['Exam 2'], s=15, c='b', marker='o', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=15, c='r', marker='x', label='Not Admitted')

plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()
'''

# Plot Sigmoid
'''
nums = np.arange(-10, 10, step=1)
plt.plot(nums, sigmoid(nums))
plt.show()
'''

# Convert DataFrames to ndarrays

df.insert(0, 'Ones', 1)

cols = df.shape[1]
x = df.iloc[:, 0:cols-1]
y = df.iloc[:, cols-1:cols]

x = x.values
y = y.values

# Map Features
x = map_feature(x[:, 0], x[:, 1])

# Create Theta
theta = np.array([[0 for _ in range(x.shape[1])]])
print(x.shape, theta.shape, y.shape)

# Set Hyperparameter
hyper_p = 1


# Minimize Function
options = {'maxiter': 100}
res = opt.minimize(log_cost, theta, (x, y, hyper_p), jac=True, method='TNC', options=options)

cost = res.fun
print(cost)
new_theta = res.x
print(new_theta)

# Determine Accuracy
accuracy = predict(new_theta, x, y)

print(f"Accuracy = {accuracy * 100}%")

