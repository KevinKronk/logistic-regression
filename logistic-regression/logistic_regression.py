import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sigmoid import sigmoid
from cost import log_cost
from gradient_descent import gradient_descent
import scipy.optimize as opt
from predict import predict

# Load Data

filename = 'ex2data1.txt'
df = pd.read_csv(filename, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


# Plot Data
'''
positive = df[df['Admitted'].isin([1])]
negative = df[df['Admitted'].isin([0])]

plt.scatter(positive['Exam 1'], positive['Exam 2'], s=15, c='b', marker='*', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=10, c='r', marker='^', label='Not Admitted')

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

theta = np.array([[0 for _ in range(cols-1)]])
print(x.shape, theta.shape, y.shape)

# Logistic Cost Function

cost = log_cost(theta, x, y)
print(cost)

alpha = 0.0009
iterations = 1000000

# new_theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
# print(new_theta)
# print(cost_history)


result = opt.fmin_tnc(func=log_cost, x0=theta, fprime=gradient_descent,
                      args=(x, y))
print(log_cost(result[0], x, y))


# plt.plot([_ for _ in range(iterations)], cost_history)
# plt.show()

opt_theta = np.reshape(result[0], (1, 3))
predictions = predict(opt_theta, x)

print(predictions)
