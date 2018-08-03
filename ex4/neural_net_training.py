"""
implementation of a neural network, trained with backpropagation
"""

##########################################################################################################
# IMPORT PACKAGES AND PREPARE DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# needed to load data (as np arrays)
from scipy.io import loadmat
data = loadmat('ex4data1.mat')
weights = loadmat('ex4weights.mat')

# weights
theta1 = weights.get('Theta1')  # theta1: weights from 1 -> 2
theta2 = weights.get('Theta2')  # theta2: weights from 2 -> 3

# X and y
X = data['X']
old_y = y = data['y'].ravel()   # to make y into an array of 1-hot, 10-element arrays
new_y = np.zeros((5000, 10))
for i in range(len(y)):
    new_y[i][y[i] - 1] = 1      # replace each element w a 10 element array
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
    plt.clf()
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
    # reshape 1D theta vector in order to make predictions
    temp_theta1 = np.reshape(theta[:10025], (25, 401))
    temp_theta2 = np.reshape(theta[10025:], (10, 26))
    # start w unregularized
    theta_cost = (1 / m) * np.sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))
    # include regularization term for all parameters EXCEPT first column of theta1, theta 2
    theta_cost += (LAMBDA / m) * (np.sum(temp_theta1[:, 1:]) + np.sum(temp_theta2[:, 1:]))
    return theta_cost


def backpropagation(theta):
    """
    given theta as vector, returns gradient as a vector: elements are gradients for matrices theta1, theta2
    """
    # gradients for theta: temp_theta1 and temp_theta2
    big_delta1 = np.zeros((25, 401))
    big_delta2 = np.zeros((10, 26))
    # iterate thru all training examples -- uses all data to calculate gradients
    for i in range(m):
        temp_theta1 = np.reshape(theta[:10025], (25, 401))
        temp_theta2 = np.reshape(theta[10025:], (10, 26))
        # 1
        inputs = np.concatenate((np.ones(1), X[i]))
        hidden_unsigmoided = np.concatenate((np.ones(1), np.matmul(temp_theta1, inputs.T)))     # z2
        hidden = sigmoid(hidden_unsigmoided)                                                    # a2
        outputs_unsigmoided = np.matmul(temp_theta2, hidden)                                    # z3
        outputs = sigmoid(outputs_unsigmoided)                                                  # a3
        # 2
        d3 = outputs - y[i]
        # 3
        d2 = np.matmul(temp_theta2.T, d3) * sigmoid_gradient(hidden_unsigmoided)
        # 4
        big_delta1 += np.matmul(d2[1:].reshape(25, 1), inputs.reshape(401, 1).T)
        big_delta2 += np.matmul(d3.reshape(10, 1), hidden.reshape(26, 1).T)
    # 5
    return (1 / m) * np.concatenate((big_delta1.ravel(), big_delta2.ravel()))


def gradient_check(theta, index):
    """
    returns gradient given theta and index of element in full theta array
    """
    # need epsilon (small number) in vector form
    epsilon = 0.00000001
    new = np.zeros(10285)
    new[index] = epsilon
    # point slightly to right
    theta_plus = theta + new
    # point slightly to left
    theta_minus = theta - new
    # return two-sided derivative approximation
    return (cost(theta_plus) - cost(theta_minus)) / (2 * epsilon)


def gradient_descent(theta, learning_rate=1, iterations=500):
    """
    returns optimized theta given starting theta (both in 1D, 'unraveled' array form)
    """
    # adjust theta in the direction of the negative gradient, repeat for specificed number of iterations
    for i in range(iterations):
        theta -= learning_rate * backpropagation(theta)
        # make sure that cost decreases on each iteraion
        print(cost(theta))
    return theta


def predict(index):
    """
    make a prediction given a point (index) in training data
    """
    # hypothesis: max value at index 9 for 0, 0 for 1, 1 for 2... adjusted accordingly
    h = np.argmax(hypothesis(final_theta)[index])
    if h + 1 != 10:
        print(h + 1)
    else:
        print(0)
    return None


def find_accuracy(theta):
    """
    computes classification accuracy of given parameters theta on the entire training set
    """
    number_correct = 0
    # count correct predictions
    for i in range(len(old_y)):
        if (np.argmax(hypothesis(theta)[i]) + 1) == old_y[i]:
            number_correct += 1
    print('\nAccuracy on training set: %0.3f' % (number_correct / len(old_y)), sep='')
    return None


def show_hidden(theta):
    """
    display weights from input -> hidden layer
    """
    plt.clf()
    plt.title('Hidden Layer Weights')
    plt.axis('off')
    # only makes sense to visualize weights from first layer -> hidden since 20 x 20 plots
    temp_theta1 = np.reshape(theta[:10025], (25, 401))
    # show 25 plots, one for each row in temp_theta1 matrix
    for i in range(1, 26):
        plt.figure(1).add_subplot(5, 5, i)
        # image needs to be flipped and rotated (just like digit images)
        plt.imshow(np.rot90(np.flip(np.reshape(temp_theta1[i - 1, 1:], (20, 20)), axis=1)), cmap=plt.cm.binary)
        plt.axis('off')
    plt.show()
    return None


##########################################################################################################
# TRAIN NETWORK TO MAKE PREDICTIONS
##########################################################################################################
# start w/ theta as 25 * 401 + 10 * 26 element array w random values between -.12 and .12
initial_theta = (np.random.rand(10285) * .24) - .12

# to check gradient: values should be close together
full_grad = backpropagation(initial_theta)
print('\nGradient checking. Following values should be close:', sep='')
print(gradient_check(initial_theta, 401 * 25 + 69))
print(full_grad[401 * 25 + 69])

# train model
print('\nBeginning training. Printing cost, should decrease every iteration:')
final_theta = gradient_descent(initial_theta)
print('\nNetwork is now trained!\n')

# to make predictions: predict(index)
# can check prediction visually: show_digit(index)
# ^^ both with with index from 0 -> 4999
