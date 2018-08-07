""" Building an SVM from scratch, including kernel, evaluated on dataset 1 """

"""
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


def show_data():
    """ Plots data without decision boundary """
    for i in range(m):
        if y[i] == 0:
            plt.scatter(X[i, 1], X[i, 2], color='tomato')
        elif y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='green')
    plt.title('Example Dataset 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


##########################################################################################################
# FUNCTIONS FOR KERNEL
##########################################################################################################
def gaussian_kernel(training_ex, landmark, sigma):
    """ Enter 2, 1D np arrays for data provided: x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2; should get 0.324652 """
    return np.exp(-(np.linalg.norm(training_ex - landmark) ** 2 / (2 * (sigma ** 2))))


def all_features(sigma=0.1):
    """ Returns matrix of similarity comparisons for ALL points in X """
    # to fill in f, array of features w shape (863, 863)
    f = np.zeros((X.shape[0], X.shape[0]))
    # iterate over every example twice to make every possible comparison
    for i in range(m):
        for j in range(m):
            f[i, j] = gaussian_kernel(X[i], X[j], sigma=sigma)
    return f


def fill_features(input_point, sigma=0.1):
    """ Returns vector of similarity comparisons with X for a given value input_point """
    # to fill in f, array of features w shape (863, 1)
    f = np.zeros(X.shape[0])
    # iterate over every example to compare
    for i in range(m):
        f[i] = gaussian_kernel(input_point, X[i, 1:], sigma=sigma)
    return f


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
    """ Returns cost given parameters theta (3,), inputs (?, 3) or (3,), ouputs matches inputs (?, 3) or (3,) """
    return C * sum(correct[1:] * cost1(np.matmul(theta, inputs[1:].T)) +
                   (1 - correct[1:]) * cost0(np.matmul(theta, inputs[1:].T))) + (1 / 2) * sum(theta[1:] ** 2)


def cost0_gradient(theta, inputs):
    """ Partial derviatives wrt theta of cost function for y = 0 """
    z = np.matmul(theta, inputs.T)
    temp = np.ones(inputs.shape)
    for i in range(len(z)):
        if z[i] <= -1:
            temp[i] = np.zeros(inputs.shape[1])
        else:
            temp[i] = inputs[i]
    return temp


def cost1_gradient(theta, inputs):
    """ Partial derviatives wrt theta of cost function for y = 1 """
    z = np.matmul(theta, inputs.T)
    temp = np.ones(inputs.shape)
    for i in range(len(z)):
        if z[i] >= 1:
            temp[i] = np.zeros(inputs.shape[1])
        else:
            temp[i] = -inputs[i]
    return temp


def gradient(theta, inputs, correct, C):
    """ Returns partial derivatives, specifically of cost0 and cost1, wrt theta """
    return C * (np.matmul((1 - correct), cost0_gradient(theta, inputs)) +
                np.matmul(correct, cost1_gradient(theta, inputs))) + sum(abs(theta[1:]))


def old_gradient(theta, inputs, correct, C):
    """ Returns gradient: partial derivative of cost function wrt parameters theta """
    return C * np.matmul((hypothesis(theta, inputs) - correct), inputs) + sum(np.abs(theta[1:]))


def gradient_descent(theta, inputs, correct, C, learning_rate=0.0001, iterations=1000, decay_term=1, old=False):
    """ Performs gradient descent to find optimal values for theta """
    if old:
        func = old_gradient
    else:
        func = gradient
    for i in range(iterations):
        theta -= (decay_term ** i) * learning_rate * func(theta, inputs, correct, C=C)
        print(cost(theta, inputs, correct, C=C))
    return theta


##########################################################################################################
# FUNCTIONS FOR EVALUATION
##########################################################################################################
def accuracy(theta, inputs):
    """ Determine accuracy of trained model """
    prediction = hypothesis(theta, inputs)
    correct = 0
    for i in range(len(y)):
        if prediction[i] == y[i]:
            correct += 1
    return correct / len(y)


def show_decision_boundary(theta):
    """ Plots decision boundary on top of data, must call gradient descent first """
    X1_increments = np.linspace(0, 4, 50)
    X2_increments = np.linspace(1.5, 5, 50)
    for i in X1_increments:
        for j in X2_increments:
            prediction = hypothesis(theta, fill_features(np.array([i, j])))
            if prediction == 0:
                plt.scatter(i, j, color='darksalmon')
            elif prediction == 1:
                plt.scatter(i, j, color='mediumseagreen')
    plt.title('Predictions')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


##########################################################################################################
# EXECUTE FUNCTIONS TO TRAIN AND EVALUATE
##########################################################################################################
# get all features
gaussian_features = all_features()

# create initial theta (3,) vector w random values -1 -> 1
initial_theta = 2 * np.random.rand(X.shape[0]) - 1

# optimize theta
final_theta = gradient_descent(initial_theta, gaussian_features, y, C=1000)

# find accuracy
print('\nAccuracy on training set of %0.3f\n' % accuracy(final_theta, gaussian_features), sep='')

# plot decision boundary with points on top
show_decision_boundary(final_theta)
show_data()
