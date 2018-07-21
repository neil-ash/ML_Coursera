"""
logistic regression -- parameters theta found using gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# get data
data = np.genfromtxt('ex2data1.txt', delimiter=',')
data = data.T

# plot data points: green = accepted, red = denied
for i in range(data[0].size):
    if data[2][i] == 0:
        plt.scatter(data[0][i], data[1][i], color='red')
    elif data[2][i] == 1:
        plt.scatter(data[0][i], data[1][i], color='green')
plt.title('Test Score vs Admittance')
plt.xlabel('Test Score #1')
plt.ylabel('Test Score #2')

# set up data nicely
x1 = data[0]                                    # test score 1
x2 = data[1]                                    # test score 2
x_all = np.array([np.ones(x1.size), x1, x2])    # includes 1s, x1, x2
y = data[2]                                     # admitted (1) or denied (0)
m = data[0].size                                # number of training examples


def sigmoid(z):
    """
    returns output of sigmoid for input z, should work with np arrays (vectors and matrices) element-wise, same
    dimension as input
    """
    # works for np arrays: defaults to element-wise unless specified matmul, etc.
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, inputs):
    """
    returns hypothesis for given parameters theta, inputs
    """
    return sigmoid(np.matmul(theta, inputs))


def cost(theta, x_all, y):
    """
    3 parameter setup necessary for optimization function, returns cost (scalar)
    """
    return (1 / m) * sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))


def gradient(theta, x_all, y):
    """
    3 parameter setup necessary for optimization function, returns gradient (3 element array)
    """
    return (1 / m) * np.matmul((hypothesis(theta) - y), x_all.T)




