"""
GENERALIZED learning curves for polynomial (and linear) regression
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
data = loadmat('ex5data1.mat')
# training
X = data['X']
y = data['y']
m = X.shape[0]
n = X.shape[1]
# testing
Xtest = data['Xtest']
ytest = data['ytest']
# validation
Xval = data['Xval']
yval = data['yval']


def powers_matrix(power, inputs):
    """
    returns matrix with columns as inputs to powers (first column: inputs^1, second column: inputs^2, etc.)
    """
    full_matrix = inputs
    for i in range(2, power + 1):
        full_matrix = np.hstack((full_matrix, inputs ** i))
    return full_matrix


def normalize_features(inputs):
    """
    scales feature columns by 1/std, shifts feature columns by -mean
    """
    # for all columns
    for i in range(inputs.shape[1]):
        inputs[:, i] -= np.mean(inputs[:, i])
        inputs[:, i] /= np.std(inputs[:, i])
    return inputs


def hypothesis(theta, inputs, m_):
    """
    given theta (9,) and inputs (12, 9) returns their matrix multiplication
    """
    return np.matmul(theta, inputs[:m_].T)


def cost(theta, inputs, correct, m_, LAMBDA=0):
    """
    returns cost for given parameters theta
    """
    return (1 / (2 * m_)) * sum((hypothesis(theta, inputs, m_) - correct[:m_].ravel()) ** 2) + (LAMBDA / (2 * m_)) * sum(theta[1:] ** 2)


def gradient(theta, inputs, correct, m_, LAMBDA=1):
    """
    returns gradient for given parameters theta
    """
    # gradient: for all terms
    theta_gradient = (1 / m_) * np.matmul(hypothesis(theta, inputs, m_) - correct[:m_].ravel(), inputs[:m_])
    # regularization: for all terms except theta0
    for i in range(1, inputs.shape[1]):
        theta_gradient[i] += (LAMBDA / m_) * theta[i]
    return theta_gradient


def gradient_descent(theta, inputs, correct, m_, LAMBDA=1, learning_rate=0.001, iterations=1000):
    """
    performs gradient descent and returns optimized values of theta
    """
    for i in range(iterations):
        theta = theta - learning_rate * gradient(theta, inputs, correct, m_, LAMBDA=LAMBDA)
    return theta


def learning_curves(power, LAMBDA=1):
    """
    displays learning curves
    """
    training_error_ls = []
    validation_error_ls = []
    i_ls = []
    for i in range(1, len(X) + 1):
        # X-power values for training and validation
        training_inputs = np.hstack((np.array([np.ones(12)]).T, normalize_features(powers_matrix(power, X))))
        validation_inputs = np.hstack((np.array([np.ones(21)]).T, normalize_features(powers_matrix(power, Xval))))
        # optimize theta WRT training data
        train_theta = gradient_descent(np.ones(power + 1), training_inputs, y, i, LAMBDA=LAMBDA)
        # find cost WRT training data and WRT validation, lambda=0 for these functions since evaluating, not optimizing
        training_error_ls.append(cost(train_theta, training_inputs, y, i, LAMBDA=0))
        validation_error_ls.append(cost(train_theta, validation_inputs, yval, m, LAMBDA=0))
        i_ls.append(i)
    # plot errors vs iterations
    plt.clf()
    plt.plot(i_ls, validation_error_ls, label='Cross-Validation Error', color='blue')
    plt.plot(i_ls, training_error_ls, label='Training Error', color='red')
    plt.legend()
    plt.title('Learning Curves for Polynomial of Power ' + str(power))
    plt.xlabel('Size of training set')
    plt.ylabel('Error')
    plt.show()
    return None


# to actually display curves
learning_curves(3)


