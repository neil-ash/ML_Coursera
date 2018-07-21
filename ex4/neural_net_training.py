"""
implementation of a neural network, trained with backpropagation
"""

##########################################################################################################
# IMPORT PACKAGES AND PREPARE DATA
##########################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# needed to load data (as np arrays)
from scipy.io import loadmat
data = loadmat('ex4data1.mat')
weights = loadmat('ex4weights.mat')

# weights
theta1 = weights.get('Theta1')      # theta1: weights from 1 -> 2
theta2 = weights.get('Theta2')      # theta2: weights from 2 -> 3

# X and y
X = data['X']
old_y = y = data['y'].ravel()   # to make y into an array of 1-hot, 10-element arrays
new_y = np.zeros((5000, 10))
for i in range(len(y)):
    new_y[i][y[i] - 1] = 1          # replace each element w a 10 element array
y = new_y

# m and n
m = X.shape[0]                  # number of training examples = 5000
n = X.shape[1]                  # number of features = 400


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def show_digit(index):
    """
    displays handwritten digit given its index
    """
    plt.title('Digit Example')
    plt.axis('off')
    # digit needs to be flipped and rotated
    plt.imshow(np.rot90(np.flip(np.reshape(X[index], (20, 20)), axis=1)), cmap=plt.cm.binary)
    plt.show()
    return None


def sigmoid(z):
    """
    returns output of sigmoid for input z (scalar, or array same dimension as input), computes np arrays element-wise
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """
    returns derivative (gradient) of sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


def given_hypothesis():
    """
    returns hypothesis for all training examples in X (shape of (5000, 10)) WRT THETA1 AND THETA2
    """
    # add column of 1s
    inputs = np.hstack((np.ones((5000, 1)), X))
    # multiply theta1 w/ inputs, then add row of 1s
    hidden = sigmoid(np.matmul(theta1, inputs.T))
    hidden = np.vstack((np.ones((1, 5000)), hidden))
    # multiply theta2 w/ hidden layer
    outputs = sigmoid(np.matmul(theta2, hidden))
    # return output nicely
    return outputs.T


def hypothesis(theta):
    """
    returns hypothesis for all training examples in X (shape of (5000, 10)) WRT INPUTTED THETA
    """
    temp_theta1 = np.reshape(theta[:10025], (25, 401))
    temp_theta2 = np.reshape(theta[10025:], (10, 26))
    # add column of 1s
    inputs = np.hstack((np.ones((5000, 1)), X))
    # multiply theta1 w/ inputs, then add row of 1s
    hidden = sigmoid(np.matmul(temp_theta1, inputs.T))
    hidden = np.vstack((np.ones((1, 5000)), hidden))
    # multiply theta2 w/ hidden layer
    outputs = sigmoid(np.matmul(temp_theta2, hidden))
    # return output nicely
    return outputs.T


def given_cost(LAMBDA=1):
    """
    returns REGULARIZED cost (scalar) WRT THETA1 AND THETA2
    """
    # start w unregularized
    theta_cost = (1 / m) * np.sum((-y) * np.log(given_hypothesis()) - (1 - y) * np.log(1 - given_hypothesis()))

    # include regularization term for all parameters EXCEPT first column of theta1, theta 2
    theta_cost += (LAMBDA / (2 * m)) * (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))

    return theta_cost


def cost(theta, LAMBDA=1):
    """
    returns REGULARIZED cost (scalar) WRT INPUTTED THETA
    """
    temp_theta1 = np.reshape(theta[:10025], (25, 401))
    temp_theta2 = np.reshape(theta[10025:], (10, 26))

    # start w unregularized
    theta_cost = (1 / m) * np.sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))

    # include regularization term for all parameters EXCEPT first column of theta1, theta 2
    theta_cost += (LAMBDA / (2 * m)) * (np.sum(temp_theta1[:, 1:] ** 2) + np.sum(temp_theta2[:, 1:] ** 2))

    return theta_cost


def backpropagation(theta):
    """

    """
    big_delta = np.zeros(25)
    # iterate thru all training examples
    for i in range(m):
        temp_theta1 = np.reshape(theta[:10025], (25, 401))
        temp_theta2 = np.reshape(theta[10025:], (10, 26))
        # 1
        inputs = np.concatenate((np.ones(1), X[i]))
        hidden_unsigmioded = np.concatenate((np.ones(1), np.matmul(temp_theta1, inputs.T)))
        hidden = sigmoid(hidden_unsigmioded)
        outputs = sigmoid(np.matmul(temp_theta2, hidden))
        # 2
        d3 = outputs - y[i]
        # 3
        d2 = np.matmul(temp_theta2.T, d3) * sigmoid_gradient(hidden_unsigmioded)
        # 4
        """ FIX THIS -- FUCKED UP """
        big_delta += d2[1:]
    # 5
    return (1 / m) * big_delta


def gradient_check(theta, index):
    """
    returns gradient given theta and index
    """
    epsilon = 0.00000001
    new = np.zeros(10285)
    new[index] = epsilon

    theta_plus = theta + new

    theta_minus = theta - new

    return (cost(theta_plus) - cost(theta_minus)) / (2 * epsilon)


# random values between -.12 and .12
initial_theta = (np.random.rand(10285) * .24) - .12
print(gradient_check(initial_theta, 0))
print(backpropagation(initial_theta)[0])


