"""
display learning curves
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


def curves_hypothesis(theta, m_, inputs=X):
    """
    given 1D array parameters theta and 2D inputs array (default X) returns matrix multiplication (hypothesis)
    """
    return np.matmul(theta, np.hstack((np.ones((int(m_), 1)), inputs[:m_])).T)


def curves_cost(theta, m_, LAMBDA=0):
    """
    returns cost for given parameters theta
    """
    return (1 / (2 * m_)) * sum((curves_hypothesis(theta, m_) - y[:m_].ravel()) ** 2) + (LAMBDA / (2 * m_)) * sum(theta[1:] ** 2)


def curves_CV_cost(theta):
    """
    returns cost for cross-validation data given parameters theta
    """
    m_ = len(Xval)
    return (1 / (2 * m_)) * sum((curves_hypothesis(theta, m_, inputs=Xval) - yval.ravel()) ** 2)


def curves_gradient(theta, m_, LAMBDA=0):
    """
    returns gradient for given parameters theta
    """
    # X0: column of ones
    theta0_gradient = (1 / m_) * sum((curves_hypothesis(theta, m_) - y[:m_].ravel()) * np.ones(m_))
    # X1: data given as X
    theta1_gradient = ((1 / m_) * sum((curves_hypothesis(theta, m_) - y[:m_].ravel()) * X[:m_].ravel())) + (LAMBDA / m_) * theta[1]
    return np.array([theta0_gradient, theta1_gradient])


def curves_gradient_descent(theta, m_, learning_rate=0.001, iterations=1000):
    """
    performs gradient descent and returns optimized values of theta
    """
    for i in range(iterations):
        theta = theta - learning_rate * curves_gradient(theta, m_)
    return theta


def learning_curves():
    """
    displays learning curves
    """
    train_ls = []
    CV_ls = []
    i_ls = []
    # iterate thru training examples
    for i in range(1, len(X) + 1):
        start_theta = np.array([1, 1])
        end_theta = curves_gradient_descent(start_theta, i)
        train_ls.append(curves_cost(end_theta, i))
        CV_ls.append(curves_CV_cost(end_theta))
        i_ls.append(i)
    # plot errors vs iterations
    plt.title('Learning Curves')
    plt.xlabel('Size of training set')
    plt.ylabel('Error')
    plt.plot(i_ls, CV_ls, label='Cross-Validation Error', color='blue')
    plt.plot(i_ls, train_ls, label='Training Error', color='red')
    plt.legend()
    plt.show()
    return None


# to actually display curves
learning_curves()


