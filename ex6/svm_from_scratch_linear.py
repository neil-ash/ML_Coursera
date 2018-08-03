"""
Building a linear SVM from scratch

Notes)
- dont force matrix shapes to be hypothesis = theta.T, X
- X: (51, 3), y: (51,), theta: (3,)
- added abs() to regularization term in cost function
"""

##########################################################################################################
# IMPORT PACKAGES AND SETUP DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
data = loadmat('ex6data1.mat')

# setup data
X = data['X']
y = data['y']
m = X.shape[0]
n = X.shape[1]

# add column of 1s to X to account for x0 term, make y 1D
X = np.hstack((np.ones((X.shape[0], 1)), X))
y = y.ravel()

# create initial theta (3,) vector w random values -0.5 -> 0.5
initial_theta = np.random.rand(3) - 0.5


def show_data():
    """ Plots data without decision boundary """
    for i in range(m):
        if y[i] == 0:
            plt.scatter(X[i, 1], X[i, 2], color='red')
        elif y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='green')
    plt.title('Example Dataset 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


##########################################################################################################
# FUNCTIONS FOR STRUCTURE AND TRAINING
##########################################################################################################
def hypothesis(theta, inputs):
    """ Returns hypothesis (0 or 1) when given parameters theta (3,) and inputs (?, 3) or (3,) """
    return (np.matmul(theta, inputs.T) >= 0).astype(int)


def cost0(z):
    """ Returns cost when y = 0 for a given z, pass in np.matmul(theta, inputs.T) """
    for i in range(len(z)):
        if z[i] <= -1:
            z[i] = 0
        else:
            z[i] = 1 + z[i]
    return z


def cost1(z):
    """ Returns cost when y = 1 for a given z, pass in np.matmul(theta, inputs.T) """
    for i in range(len(z)):
        if z[i] >= 1:
            z[i] = 0
        else:
            z[i] = 1 - z[i]
    return z


def cost(theta, inputs, correct, C):
    """ Returns cost given parameters theta (3,), inputs (?, 3) or (3,), ouputs matching inputs (?, 3) or (3,), and hyperparameter C """
    return C * sum(correct[1:] * cost1(np.matmul(theta, inputs[1:].T)) + (1 - correct[1:]) * cost0(np.matmul(theta, inputs[1:].T))) \
           + (1 / 2) * sum(theta[1:] ** 2)


def cost0_gradient(theta, inputs):
    """ Partial derviatives wrt theta of cost function for y = 0 """
    z = np.matmul(theta, inputs.T)
    temp = np.ones(inputs.shape)
    for i in range(len(z)):
        if z[i] <= -1:
            temp[i] = np.zeros(3)
        else:
            temp[i] = inputs[i]
    return temp


def cost1_gradient(theta, inputs):
    """ Partial derviatives wrt theta of cost function for y = 1 """
    z = np.matmul(theta, inputs.T)
    temp = np.ones(inputs.shape)
    for i in range(len(z)):
        if z[i] >= 1:
            temp[i] = np.zeros(3)
        else:
            temp[i] = -inputs[i]
    return temp


def gradient(theta, inputs, correct, C):
    """ Returns partial derivatives, specifically of cost0 and cost1, wrt theta """
    return C * (np.matmul((1 - correct), cost0_gradient(theta, inputs)) + np.matmul(correct, cost1_gradient(theta, inputs))) \
           + sum(abs(theta[1:]))


def old_gradient(theta, inputs, correct, C):
    """ Returns gradient: partial derivative of cost function wrt parameters theta """
    return C * np.matmul((hypothesis(theta, inputs) - correct), inputs) + sum(np.abs(theta[1:]))


def gradient_descent(theta, inputs, correct, C, old=False, learning_rate=0.0001, iterations=10000):
    """ Performs gradient descent to find optimal values for theta """
    if old:
        func = old_gradient
    else:
        func = gradient
    for i in range(iterations):
        theta -= learning_rate * func(theta, inputs, correct, C=C)
        print(cost(theta, X, y, C=C))
    return theta


##########################################################################################################
# EVALUATE MODEL
##########################################################################################################
def show_decision_boundary(theta):
    """ Plots decision boundary on top of data, must call gradient descent first """
    show_data()
    ptX1 = min(X[:, 1]), max(X[:, 1])
    ptX2 = (-theta[0] - theta[1] * ptX1[0]) / theta[2], (-theta[0] - theta[1] * ptX1[1]) / theta[2]
    plt.plot(ptX1, ptX2, color='black')
    plt.show()
    return None


final_theta = gradient_descent(initial_theta, X, y, C=1)
show_decision_boundary(final_theta)


'''
To debug easily: 

theta = initial_theta
inputs = X
correct = y
cost(initial_theta, X, y)
gradient(initial_theta, X, y)
gradient_descent(initial_theta, X, y)
'''
