"""
regularized linear regression
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

# plot training data
plt.title('Training Data')
plt.xlabel('Change in water level')
plt.ylabel('Water leaving dam')
plt.scatter(X, y, color='skyblue')


def hypothesis(theta, inputs=X):
    """
    given 1D array parameters theta and 2D inputs array (default X) returns matrix multiplication (hypothesis)
    """
    return np.matmul(theta, np.hstack((np.ones((len(inputs), 1)), inputs)).T)


def cost(theta, LAMBDA=0):
    """
    returns cost for given parameters theta
    """
    return (1 / (2 * m)) * sum((hypothesis(theta) - y.ravel()) ** 2) + (LAMBDA / (2 * m)) * sum(theta[1:] ** 2)


def gradient(theta, LAMBDA=0):
    """
    returns gradient for given parameters theta
    """
    # X0: column of ones
    theta0_gradient = (1 / m) * sum((hypothesis(theta) - y.ravel()) * np.ones(len(y)))
    # X1: data given as X
    theta1_gradient = ((1 / m) * sum((hypothesis(theta) - y.ravel()) * X.ravel())) + (LAMBDA / m) * theta[1]
    return np.array([theta0_gradient, theta1_gradient])


def gradient_descent(theta, learning_rate=0.001, iterations=20000):
    """
    performs gradient descent and returns optimized values of theta
    """
    for i in range(iterations):
        theta = theta - learning_rate * gradient(theta)
        # print(cost(theta))
    return theta


# optimize theta using gradient descent
initial_theta = np.array([1, 1])
final_theta = gradient_descent(initial_theta)

# plot line of best fit
pt_X = min(X)[0], max(X)[0]
pt_y = hypothesis(final_theta, np.array([min(X)]))[0], hypothesis(final_theta, np.array([max(X)]))[0]
plt.plot(pt_X, pt_y, color='tomato')
plt.show()

