import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sigmoid import sigmoid
from cost import log_cost

# Load Data

filename = 'ex2data1.txt'
df = pd.read_csv(filename, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


# Plot Data

positive = df[df['Admitted'].isin([1])]
negative = df[df['Admitted'].isin([0])]

plt.scatter(positive['Exam 1'], positive['Exam 2'], s=15, c='b', marker='*', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=10, c='r', marker='^', label='Not Admitted')

plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()


# Plot Sigmoid

nums = np.arange(-10, 10, step=1)
plt.plot(nums, sigmoid(nums))
plt.show()


# Convert DataFrames to ndarrays

df.insert(0, 'Ones', 1)

cols = df.shape[1]
x = df.iloc[:, 0:cols-1]
y = df.iloc[:, cols-1:cols]

x = x.values
y = y.values

theta = np.array([[0 for _ in range(cols-1)]])


# Logistic Cost Function

cost = log_cost(x, y, theta)
print(cost)

