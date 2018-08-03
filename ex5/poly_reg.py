"""
polynomial regression
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


def powers_matrix(power, inputs=X):
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


def hypothesis(theta, inputs):
    """
    given theta (9,) and inputs (12, 9) returns their matrix multiplication
    """
    return np.matmul(theta, inputs.T)


def cost(theta, inputs, correct=y, LAMBDA=0):
    """
    returns cost for given parameters theta (9,) with given inputs (12, 9)
    """
    return (1 / (2 * m)) * sum((hypothesis(theta, inputs) - correct.ravel()) ** 2) + (LAMBDA / (2 * m)) * sum(theta[1:] ** 2)


def gradient(theta, inputs, correct=y, LAMBDA=1):
    """
    finds gradient given theta (9,) and inputs (12, 9)
    """
    # gradient: for all terms
    theta_gradient = (1 / m) * np.matmul(hypothesis(theta, inputs) - correct.ravel(), inputs)
    # regularization: for all terms except theta0
    for i in range(1, inputs.shape[1]):
        theta_gradient[i] += (LAMBDA / m) * theta[i]
    return theta_gradient


def gradient_descent(theta, inputs, correct=y, LAMBDA=1, learning_rate=0.01, iterations=10000):
    """
    given theta (9,), performs gradient descent and returns optimized values of theta
    """
    for i in range(iterations):
        theta = theta - learning_rate * gradient(theta, inputs, correct=correct, LAMBDA=LAMBDA)
    return theta


def choosing_lambda(order):
    """
    returns best value of lambda (lambda with lowest error on cross validation set), displays plots for visualization
    """
    lambda_ls = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    validation_error_ls = []
    training_error_ls = []
    # test reasonable lambda values
    for i in lambda_ls:
        # X-power values for validation and training
        validation_inputs = np.hstack((np.array([np.ones(21)]).T, normalize_features(powers_matrix(order, inputs=Xval))))
        training_inputs = np.hstack((np.array([np.ones(12)]).T, normalize_features(powers_matrix(order, inputs=X))))
        # run gradient descent to train with training inputs, lambda value to test, initial theta as 9 1s
        train_theta = gradient_descent(np.ones(order + 1), training_inputs, correct=y, LAMBDA=i)
        # find training error and validation error
        training_error_ls.append(cost(train_theta, training_inputs, correct=y, LAMBDA=0))
        validation_error_ls.append(cost(train_theta, validation_inputs, correct=yval, LAMBDA=0))
    # setup figure 2, clear previous figure 2 plot
    plt.figure(2)
    plt.clf()
    # plot training error and validation errors vs lambda
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
    """
    returns best power (order of polynomial) resulting in the lowest validation error, shows visualization plots
    """
    power_ls = [i for i in range(1, 26)]
    validation_error_ls = []
    training_error_ls = []
    # iterates thru powers 1 -> 25
    for power in range(1, 26):
        # inputs: X to the power i in matrix for training, Xtest to the power i for validation
        training_inputs = np.hstack((np.array([np.ones(12)]).T, normalize_features(powers_matrix(power, inputs=X))))
        validation_inputs = np.hstack((np.array([np.ones(21)]).T, normalize_features(powers_matrix(power, inputs=Xval))))
        # train theta on training data -- this lambda needs to be 0, or else a higher order polynomial will ~always be better
        trained_theta = gradient_descent(np.ones(power + 1), training_inputs, correct=y, LAMBDA=0)
        # evaluate vs training data and vs validation data
        training_error_ls.append(cost(trained_theta, training_inputs, correct=y, LAMBDA=0))
        validation_error_ls.append(cost(trained_theta, validation_inputs, correct=yval, LAMBDA=0))
    # plot training and validation errors vs power
    plt.figure(3)
    plt.clf()
    plt.plot(power_ls, validation_error_ls, color='green', label='Cross Validation')
    plt.plot(power_ls, training_error_ls, color='blue', label='Train')
    plt.legend()
    plt.title('Error vs Polynomial Order')
    plt.xlabel('Polynomial Order (highest power)')
    plt.ylabel('Error')
    best_power = power_ls[validation_error_ls.index(min(validation_error_ls))]
    print('\nMinimum cross validation error occurs with polynomial of order', best_power)
    return best_power


def show_regression(order, LAMBDA=1):
    """
    show regression curve for given lambda
    """
    # X-power values for validation and training
    training_inputs = np.hstack((np.array([np.ones(12)]).T, normalize_features(powers_matrix(order, inputs=X))))
    # run gradient descent
    end_theta = gradient_descent(np.ones(order + 1), training_inputs, LAMBDA=LAMBDA)
    # setup figure 1, clear previous figure 1 plot
    plt.figure(1)
    plt.clf()
    # plot training data and trend line
    plt.scatter(X, y, color='tomato', label='Acutal')
    plt.plot(sorted(X), sorted(hypothesis(end_theta, training_inputs)), color='skyblue', label='Predictions')
    plt.legend()
    plt.title('Training Data and Regression Curve')
    plt.xlabel('Change in water level')
    plt.ylabel('Water leaving dam')
    plt.show()
    return None


def test_results(order, LAMBDA=1):
    """
    prints cost on test data for given lambda
    """
    # X-power values for test and training
    training_inputs = np.hstack((np.array([np.ones(12)]).T, normalize_features(powers_matrix(order, inputs=X))))
    test_inputs = np.hstack((np.array([np.ones(21)]).T, normalize_features(powers_matrix(order, inputs=Xtest))))
    # optimize theta by training on training data
    train_theta = gradient_descent(np.ones(order + 1), training_inputs, y, LAMBDA=LAMBDA)
    # display cost WRT test data
    print('\nCost on test data with lambda = %d: %0.2f\n' % (LAMBDA, cost(train_theta, test_inputs, correct=ytest, LAMBDA=0)))
    return None


# find the best polynomial order, value of lambda, then plot curve and show results
use_power = choosing_power()
use_lambda = choosing_lambda(use_power)
show_regression(use_power, LAMBDA=use_lambda)
test_results(use_power, LAMBDA=use_lambda)







