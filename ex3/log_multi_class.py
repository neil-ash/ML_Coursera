""" Multiclass logistic regression for MNIST dataset """

##########################################################################################################
# IMPORT PACKAGES AND PREPARE DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

# needed to load data (as np arrays)
from scipy.io import loadmat
data = loadmat('ex3data1.mat')
X = data['X']

# to make y into an array of 1-hot, 10-element arrays
old_y = y = data['y'].ravel()
new_y = np.zeros((5000, 10))
for i in range(len(y)):
    # convert 10 to 0
    if y[i] == 10:
        y[i] = 0
    # replace each element w a 10 element, 1-hot array
    new_y[i, y[i]] = 1
y = new_y

# number of training examples (5000), number of features (400)
m = X.shape[0]
n = X.shape[1]


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def show_digit(index):
    """ Displays singl handwritten digit given its index """
    plt.title('Digit Example')
    plt.axis('off')
    # digit needs to be flipped and rotated
    plt.imshow(np.rot90(np.flip(np.reshape(X[index], (20, 20)), axis=1)), cmap=plt.cm.binary)
    plt.show()
    return None


def sigmoid(z):
    """ Returns output of sigmoid for input z, works with np arrays element-wise """
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, inputs):
    """ Returns hypothesis (array, size is number of inputs) for given parameters theta and inputs """
    return sigmoid(np.matmul(theta, inputs.T))


def cost(theta, digit, LAMBDA=1):
    """ Returns cost function (single value) for given parameters theta """
    # start w unregularized
    theta_cost = (1 / m) * np.sum((-y[:, digit]) * np.log(hypothesis(theta, X))
                                  - (1 - y[:, digit]) * np.log(1 - hypothesis(theta, X)))
    # include regularization term for all parameters EXCEPT theta[0]
    theta_cost += (LAMBDA / (2 * m)) * np.sum(theta[1:] ** 2)
    return theta_cost


def gradient(theta, digit, LAMBDA=1):
    """ Returns gradient of cost function for given parameters theta """
    # start w unregularized
    theta_gradient = (1 / m) * np.matmul((hypothesis(theta, X) - y[:, digit]), X)
    # include regularization term for all parameters EXCEPT theta[0]
    theta_gradient[1:] += (LAMBDA / m) * theta[1:]
    return theta_gradient


def gradient_descent(theta, digit, learning_rate=0.1, iterations=1000):
    """ Performs gradient descent, returns optimal values of theta """
    for i in range(iterations):
        theta -= learning_rate * gradient(theta, digit)
    return theta


def digit_specific_accuracy(digit, learning_rate=1, iterations=2000):
    """ Prints accuracy in classifying a given digit, returns parameters theta used to make classification """
    # find best values for parameters theta using gradient descent
    theta = gradient_descent(np.zeros(400), digit, learning_rate=learning_rate, iterations=iterations)
    # to determine number of correct classifications
    correct = 0
    for i in range(m):
        if hypothesis(theta, X[i]) >= 0.5 and old_y[i] == digit:
            correct += 1
        elif hypothesis(theta, X[i]) < 0.5 and old_y[i] != digit:
            correct += 1
    print('Accuracy on binary classification of digit %0.0f: %0.3f' % (digit, correct / m))
    return theta


def train_all(learning_rate=1, iterations=2000):
    """ Returns list of ALL trained parameters theta """
    theta_ls = [np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10),
                np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(10)]
    for i in range(len(theta_ls)):
        theta_ls[i] = digit_specific_accuracy(i, learning_rate=learning_rate, iterations=iterations)
    correct = 0
    for i in range(m):
        temp_ls = []
        for j in range(10):
            temp_ls.append(hypothesis(theta_ls[j], X[i]))
        if temp_ls.index(max(temp_ls)) == int(i / int(m / 10)):
            correct += 1
    print('\nOverall accuracy: %0.3f\n' % (correct / m), sep='')
    return np.asarray(theta_ls)


def predict(theta_ls, index):
    """ Make a prediction on a given digit """
    # prediction w every classifier
    temp_ls = []
    for i in range(10):
        temp_ls.append(hypothesis(theta_ls[i], X[index]))
    # display digit
    show_digit(index)
    # actual prediction: index of largest (most confident) prediction
    print('Prediction: %0.0d, with confidence %0.3f' % (temp_ls.index(max(temp_ls)), max(temp_ls)))
    return None


##########################################################################################################
# TRAIN NETWORK AND MAKE PREDICTIONS
##########################################################################################################
trained_theta = train_all(learning_rate=1, iterations=2000)
predict(trained_theta, 4000)

