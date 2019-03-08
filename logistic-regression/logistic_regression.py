import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# exam 1 score, exam 2 score, admitted?

filename = 'ex2data1.txt'
df = pd.read_csv(filename, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


positive = df[df['Admitted'].isin([1])]
negative = df[df['Admitted'].isin([0])]

plt.scatter(positive['Exam 1'], positive['Exam 2'], s=15, c='b', marker='*', label='Admitted')
plt.scatter(negative['Exam 1'], negative['Exam 2'], s=10, c='r', marker='^', label='Not Admitted')

plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step=1)
plt.plot(nums, sigmoid(nums))
plt.show()

