"""
logistic regression -- parameters theta found using gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

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
x1 = data[0]                                    # test score 1
x2 = data[1]                                    # test score 2
x_all = np.array([np.ones(x1.size), x1, x2])    # includes 1s, x1, x2
y = data[2]                                     # admitted (1) or denied (0)
m = data[0].size                                # number of training examples


def sigmoid(z):
    """
    returns output of sigmoid for input z, should work with np arrays (vectors and matrices) element-wise, same
    dimension as input
    """
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, inputs):
    """
    returns hypothesis for given parameters theta (3 element array), inputs (3 rows, any columns)
    """
    return sigmoid(np.matmul(theta, inputs))


def cost(theta):
    """
    returns cost function for given parameters theta
    """
    return (1 / m) * sum((-y) * np.log(hypothesis(theta, x_all)) - (1 - y) * np.log(1 - hypothesis(theta, x_all)))


def gradient(theta):
    """
    returns gradient (3 element array) of cost function for given parameters theta
    """
    return (1 / m) * np.matmul((hypothesis(theta, x_all) - y), x_all.T)


def gradient_descent(theta, learning_rate=0.003, iterations=2000000):
    """
    performs gradient descent, returns optimal values of theta
    """
    for i in range(iterations):
        theta -= learning_rate * gradient(theta)
        #print(cost(theta))
    return theta


# find the best values of theta using gradient descent
initial_theta = np.zeros(3)
final_theta = gradient_descent(initial_theta)
print('\nBest values for theta: %0.2f, %0.2f, %0.2f' % (final_theta[0], final_theta[1], final_theta[2]), sep='')

# plot boundary condition -- write as x2 = ...
show_x1 = [min(x1), max(x1)]
show_x2 = [(-final_theta[0] - final_theta[1] * min(x1)) / final_theta[2], (-final_theta[0] - final_theta[1] * max(x1)) / final_theta[2]]
plt.plot(show_x1, show_x2, color='black')
plt.show()

# make a prediction on never-before-seen input
print('Prediction on student with Exam 1 score of 45, Exam 2 score of 85: %0.2f' % hypothesis(final_theta, np.array([1, 45, 85])))
print('\n', end='')

