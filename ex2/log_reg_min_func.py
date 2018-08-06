""" Logistic regression using sicpy minimization function """

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


########################################################################################################################
# LOAD, PLOT, AND FORMAT DATA
########################################################################################################################
# get data
data = np.genfromtxt('ex2data1.txt', delimiter=',')
data = data.T

# plot data points: green = accepted, red = denied
for i in range(data[0].size):
    if data[2][i] == 0:
        plt.scatter(data[0][i], data[1][i], color='red')
    elif data[2][i] == 1:
        plt.scatter(data[0][i], data[1][i], color='green')
plt.title('Test Score vs Admittance')
plt.xlabel('Test Score #1')
plt.ylabel('Test Score #2')

# set up data nicely
X1 = data[0]                                    # test score 1
X2 = data[1]                                    # test score 2
X = np.array([np.ones(X1.size), X1, X2])        # includes 1s, x1, x2
y = data[2]                                     # admitted (1) or denied (0)
m = data[0].size                                # number of training examples


########################################################################################################################
# FUNCTIONS FOR LOGISTIC REGRESSION
########################################################################################################################
def sigmoid(z):
    """ Returns sigmoid of input, works with np arrays (vectors and matrices) element-wise, same dimension as input """
    return 1 / (1 + np.exp(-z))


def hypothesis(theta):
    """ Returns hypothesis (100 element array) FOR ALL TRAINING EXAMPLES given parameters theta """
    return sigmoid(np.matmul(theta, X))


def cost(theta, X, y):
    """ Returns total cost given theta, 3 parameter setup necessary for optimization function """
    return (1 / m) * sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))


def gradient(theta, X, y):
    """ Returns gradient given theta, 3 parameter setup necessary for optimization function """
    return (1 / m) * np.matmul((hypothesis(theta) - y), X.T)


########################################################################################################################
# FUNCTIONS FOR LOGISTIC REGRESSION
########################################################################################################################
# initialize theta as array of 3 0s
initial_theta = np.zeros(len(data))

# use optimization function to find best values for theta, returned as result[0]
result = op.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y))
final_theta = result[0]

# print the decision boundary equation, final cost
print('Decision boundary eq: %0.2f + %0.2f * x1 + %0.2f * x2 = 0' % (result[0][0], result[0][1], result[0][2]))
print('Final cost after optimization: %.2f' % cost(result[0], X, y))

# plot boundary condition -- write as x2 = ...
show_x1 = [min(X1), max(X1)]
show_x2 = [(-final_theta[0] - final_theta[1] * min(X1)) / final_theta[2],
           (-final_theta[0] - final_theta[1] * max(X1)) / final_theta[2]]
plt.plot(show_x1, show_x2, color='black')
plt.show()

# make prediction on new student: Exam 1 score of 45 and an Exam 2 score of 85
print('Prediction on student with Exam 1 score of 45, Exam 2 score of 85: %0.2f'
      % sigmoid(np.matmul(final_theta, np.array([1, 45, 85]))))

