""" Polynomial Regression, includes code to find the best order polynomial """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')


##########################################################################################################
# LOAD DATA
##########################################################################################################
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


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def powers_matrix(power, inputs):
    """ Returns matrix with columns as inputs to powers (first column: inputs^1, second column: inputs^2, etc.) """
    full_matrix = inputs
    for i in range(2, power + 1):
        full_matrix = np.hstack((full_matrix, inputs ** i))
    return full_matrix


def normalize_features(inputs):
    """ Scales feature columns by 1 / (column std), shifts feature columns by -(column mean) """
    # for all columns
    for i in range(inputs.shape[1]):
        inputs[:, i] -= np.mean(inputs[:, i])
        inputs[:, i] /= np.std(inputs[:, i])
    return inputs


def hypothesis(theta, inputs):
    """ Given theta (9,) and inputs (12, 9) returns their matrix multiplication """
    return np.matmul(theta, inputs.T)


def cost(theta, inputs, correct, LAMBDA=0):
    """ Returns regularized cost for given parameters theta (9,) with given inputs (12, 9) """
    return (1 / (2 * m)) * sum((hypothesis(theta, inputs) - correct.ravel()) ** 2) \
           + (LAMBDA / (2 * m)) * sum(theta[1:] ** 2)


def gradient(theta, inputs, correct, LAMBDA=1):
    """ Finds gradient given theta (9,) and inputs (12, 9) """
    # gradient: for all terms
    theta_gradient = (1 / m) * np.matmul(hypothesis(theta, inputs) - correct.ravel(), inputs)
    # regularization: for all terms except theta0
    theta_gradient[1:] += (LAMBDA / m) * theta[1:]
    return theta_gradient


def gradient_descent(theta, inputs, correct, LAMBDA=1, learning_rate=0.01, iterations=10000):
    """ Given theta (9,), performs gradient descent and returns optimized values of theta """
    for i in range(iterations):
        theta = theta - learning_rate * gradient(theta, inputs, correct, LAMBDA=LAMBDA)
    return theta


def find_theta(order, LAMBDA):
    """ Returns training inputs, optimal theta (from training set) for given polynomial order and lambda """
    # X-power values for validation and training
    training_inputs = np.hstack((np.ones(X.shape), normalize_features(powers_matrix(order, X))))
    # run gradient descent
    train_theta = gradient_descent(np.ones(order + 1), training_inputs, y, LAMBDA=LAMBDA)
    return training_inputs, train_theta


##########################################################################################################
# FUNCTIONS FOR HYPERPARAMETER SELECTION
##########################################################################################################
def choosing_lambda(order):
    """ Returns best value of lambda (lambda with lowest error on cross validation set), displays visualization """
    lambda_ls = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    validation_error_ls = []
    training_error_ls = []
    # test reasonable lambda values
    for LAMBDA in lambda_ls:
        # train on traning data, testing various lambda values
        training_inputs, trained_theta = find_theta(order, LAMBDA)
        # evaluate on validation
        validation_inputs = np.hstack((np.ones(Xval.shape), normalize_features(powers_matrix(order, Xval))))
        # find training error and validation error w/ out regularization
        training_error_ls.append(cost(trained_theta, training_inputs, y, LAMBDA=0))
        validation_error_ls.append(cost(trained_theta, validation_inputs, yval, LAMBDA=0))
    # plot training error and validation errors vs lambda
    plt.figure(2)
    plt.clf()
    plt.plot(lambda_ls, validation_error_ls, color='green', label='Cross Validation')
    plt.plot(lambda_ls, training_error_ls, color='blue', label='Train')
    plt.legend()
    plt.title('Error vs Regularization Parameter')
    plt.xlabel('lambda')
    plt.ylabel('Error (without regularization term)')
    plt.show()
    # print and return best value of lambda
    best_lambda = lambda_ls[validation_error_ls.index(min(validation_error_ls))]
    print('\nMinimum cross validation error occurs at lambda = ', best_lambda, sep='')
    return best_lambda


def choosing_power():
    """ Returns best power (order of polynomial) resulting in the lowest validation error, shows visualization """
    power_ls = [i for i in range(1, 26)]
    validation_error_ls = []
    training_error_ls = []
    # iterates thru powers 1 -> 25
    for power in range(1, 26):
        # train w/ training data, this lambda needs to be 0 or else higher order polynomial will ~always be better
        training_inputs, trained_theta = find_theta(power, LAMBDA=0)
        # evaluate on validation
        validation_inputs = np.hstack((np.ones(Xval.shape), normalize_features(powers_matrix(power, Xval))))
        # find training error and validation error w/ out regularization
        training_error_ls.append(cost(trained_theta, training_inputs, y, LAMBDA=0))
        validation_error_ls.append(cost(trained_theta, validation_inputs, yval, LAMBDA=0))
    # plot training and validation errors vs power
    plt.figure(3)
    plt.clf()
    plt.plot(power_ls, validation_error_ls, color='green', label='Cross Validation')
    plt.plot(power_ls, training_error_ls, color='blue', label='Train')
    plt.legend()
    plt.title('Error vs Polynomial Order')
    plt.xlabel('Polynomial Order (highest power)')
    plt.ylabel('Error')
    # print and return best power
    best_power = power_ls[validation_error_ls.index(min(validation_error_ls))]
    print('\nMinimum cross validation error occurs with polynomial of order', best_power)
    return best_power


##########################################################################################################
# FUNCTIONS TO SEE RESULTS
##########################################################################################################
def show_regression(order, LAMBDA):
    """ Show regression curve for given polynomial order and regularization lambda """
    # train w/ training data
    training_inputs, trained_theta = find_theta(order, LAMBDA)
    # setup figure 1, clear previous figure 1 plot
    plt.figure(1)
    plt.clf()
    # plot training data and trend line
    plt.scatter(X, y, color='tomato', label='Acutal')
    plt.plot(sorted(X), sorted(hypothesis(trained_theta, training_inputs)), color='skyblue', label='Predictions')
    plt.legend()
    plt.title('Training Data and Regression Curve')
    plt.xlabel('Change in water level')
    plt.ylabel('Water leaving dam')
    plt.show()
    return None


def test_results(order, LAMBDA):
    """ Prints cost on test data for given lambda """
    # train w/ training data
    training_inputs, trained_theta = find_theta(order, LAMBDA)
    # then compare with test
    testing_inputs = np.hstack((np.ones(Xtest.shape), normalize_features(powers_matrix(order, Xtest))))
    # display cost WRT test data
    print('\nCost on test data with lambda = %d: %0.2f\n'
          % (LAMBDA, cost(trained_theta, testing_inputs, ytest, LAMBDA=0)))
    return None


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
# find the best polynomial order, value of lambda, then plot curve and show results
use_power = choosing_power()
use_lambda = choosing_lambda(use_power)
show_regression(use_power, LAMBDA=use_lambda)
test_results(use_power, LAMBDA=use_lambda)



