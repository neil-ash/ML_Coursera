""" Implementation of a neural network trained with backpropagation for the MNIST dataset (20x20 pixel images) """

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

# X and y
X = data['X']
old_y = y = data['y'].ravel()   # to make y into an array of 1-hot, 10-element arrays
new_y = np.zeros((5000, 10))
for i in range(len(y)):
    new_y[i, y[i] - 1] = 1      # replace each element w a 10 element array
y = new_y

# m and n
m = X.shape[0]                  # number of training examples = 5000
n = X.shape[1]                  # number of features = 400


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def show_digit(index):
    """ Displays handwritten digit given its index """
    plt.clf()
    plt.title('Digit Example')
    plt.axis('off')
    # digit needs to be flipped and rotated
    plt.imshow(np.rot90(np.flip(np.reshape(X[index], (20, 20)), axis=1)), cmap=plt.cm.binary)
    plt.show()
    return None


def sigmoid(z):
    """ Returns output of sigmoid for input z, works with np arrays element-wise """
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """ Returns derivative (gradient) of sigmoid function """
    return sigmoid(z) * (1 - sigmoid(z))


def hypothesis(theta):
    """ Returns hypothesis for all training examples in X (5000, 10) WRT inputted theta """
    # theta contains weights for input and hidden layers
    temp_theta1 = np.reshape(theta[:(25 * 401)], (25, 401))      # input: 400 nodes, plus bias -> hidden: 25 nodes
    temp_theta2 = np.reshape(theta[-(10 * 26):], (10, 26))       # hidden: 25 nodes, plus bias -> output: 10 nodes
    # input layer: inputs X with additional column of 1s (biases for each point in training set)
    inputs = np.hstack((np.ones((m, 1)), X))
    # hidden layer: multiply theta1 w/ inputs, then add row of 1s (biases for each point in training set)
    hidden = sigmoid(np.matmul(temp_theta1, inputs.T))
    hidden = np.vstack((np.ones((1, m)), hidden))
    # output layer: multiply theta2 w/ hidden layer
    outputs = sigmoid(np.matmul(temp_theta2, hidden))
    # return output nicely
    return outputs.T


def cost(theta, LAMBDA=1):
    """ Returns regularized cost (scalar) WRT inputted theta """
    # reshape 1D theta vector in order to make predictions
    temp_theta1 = np.reshape(theta[:(25 * 401)], (25, 401))
    temp_theta2 = np.reshape(theta[-(10 * 26):], (10, 26))
    # start w/ unregularized
    theta_cost = (1 / m) * np.sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))
    # include regularization term for all parameters except first column of theta1, theta 2
    theta_cost += (LAMBDA / m) * (np.sum(temp_theta1[:, 1:]) + np.sum(temp_theta2[:, 1:]))
    return theta_cost


def backpropagation(theta):
    """ Given theta as vector, returns gradient as a vector w/ elements as gradients for matrices theta1, theta2 """
    # gradients for theta, same shape as temp_theta1 and temp_theta2
    big_delta1 = np.zeros((25, 401))
    big_delta2 = np.zeros((10, 26))
    # iterate thru all training examples (batch gradient descent)
    for i in range(m):
        temp_theta1 = np.reshape(theta[:(25 * 401)], (25, 401))
        temp_theta2 = np.reshape(theta[-(10 * 26):], (10, 26))
        # 1. feedforward pass to compute inital activations in each layer, a = g(z) w/ g() as sigmoid
        inputs = np.concatenate((np.ones(1), X[i]))
        hidden_unsigmoided = np.concatenate((np.ones(1), np.matmul(temp_theta1, inputs.T)))     # z2
        hidden = sigmoid(hidden_unsigmoided)                                                    # a2
        outputs_unsigmoided = np.matmul(temp_theta2, hidden)                                    # z3
        outputs = sigmoid(outputs_unsigmoided)                                                  # a3
        # 2. compute difference in initial predictions and actual values y (errors in output aka layer 3 nodes)
        d3 = outputs - y[i]
        # 3. backpropagate: multiply output errors by hidden parameters (errors in hidden layer aka layer 2 nodes)
        d2 = np.matmul(temp_theta2.T, d3) * sigmoid_gradient(hidden_unsigmoided)
        # 4. accumulate the gradient: error in next layer times activation in previous
        big_delta1 += np.matmul(d2[1:].reshape(25, 1), inputs.reshape(401, 1).T)
        big_delta2 += np.matmul(d3.reshape(10, 1), hidden.reshape(26, 1).T)
    # 5. divide by number of training examples and return gradient in same shape as theta
    return (1 / m) * np.concatenate((big_delta1.ravel(), big_delta2.ravel()))


def gradient_check(theta, index):
    """ Returns gradient (computed numerically) given theta and index of element in full theta array """
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
    """ Returns optimized theta given initial theta, both in consistent 1D-array form """
    # adjust theta in the direction of the negative gradient, repeat for specificed number of iterations
    for i in range(iterations):
        theta -= learning_rate * backpropagation(theta)
        print(cost(theta))
    return theta


def predict(index):
    """ Make a prediction on a single digit (index in training data) """
    # hypothesis: max value at index 9 for 0, 0 for 1, 1 for 2... adjusted accordingly
    h = np.argmax(hypothesis(final_theta)[index])
    print(h + 1) if (h + 1 != 10) else print(0)
    return None


def find_accuracy(theta):
    """ Prints classification accuracy of given parameters theta on the entire training set """
    number_correct = 0
    # count correct predictions
    for i in range(len(old_y)):
        if (np.argmax(hypothesis(theta)[i]) + 1) == old_y[i]:
            number_correct += 1
    print('\nAccuracy on training set: %0.3f\n' % (number_correct / len(old_y)), sep='')
    return None


def show_hidden(theta):
    """ Display weights from input -> hidden layer """
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
initial_theta = (np.random.rand(25 * 401 + 10 * 26) * .24) - .12

# to check gradient: values should be close together
full_grad = backpropagation(initial_theta)
print('\nGradient checking. Following values should be close:', sep='')
print(gradient_check(initial_theta, 401 * 25 + 69))
print(full_grad[401 * 25 + 69])

# train model
print('\nBeginning training. Printing cost, should decrease every iteration:')
final_theta = gradient_descent(initial_theta)
print('\nNetwork is now trained!')

# print accuracy of trained model
find_accuracy(final_theta)

# to make a prediction and see digit (ex: on a '3'):
#       predict(1500)
#       show_digit(1500)
#
# to show hidden layer activations:
#       show_hidden(final_theta)


