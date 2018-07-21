"""
logistic regression, parameters theta found using minimization function
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

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
    # works for np arrays: defaults to element-wise unless specified matmul, etc.
    return 1 / (1 + np.exp(-z))


def hypothesis(theta):
    """
    returns hypothesis (100 element array) FOR ALL TRAINING EXAMPLES given parameters theta
    """
    return sigmoid(np.matmul(theta, x_all))


def cost(theta, x_all, y):
    """
    3 parameter setup necessary for optimization function, returns cost (scalar)
    """
    return (1 / m) * sum((-y) * np.log(hypothesis(theta)) - (1 - y) * np.log(1 - hypothesis(theta)))


def gradient(theta, x_all, y):
    """
    3 parameter setup necessary for optimization function, returns gradient (3 element array)
    """
    return (1 / m) * np.matmul((hypothesis(theta) - y), x_all.T)


# initialize theta as array of 3 0s
theta = np.zeros(len(data))

# use optimization function to find best values for theta, returned as result[0]
result = op.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x_all, y))
theta = result[0]

# print the decision boundary equation, final cost
print('Decision boundary eq: %0.2f + %0.2f * x1 + %0.2f * x2 = 0' % (result[0][0], result[0][1], result[0][2]))
print('Final cost after optimization:', cost(result[0], x_all, y))

# plot boundary condition -- write as x2 = ...
show_x1 = [min(x1), max(x1)]
show_x2 = [(-theta[0] - theta[1] * min(x1)) / theta[2], (-theta[0] - theta[1] * max(x1)) / theta[2]]
plt.plot(show_x1, show_x2, color='black')
plt.show()

# make prediction on new student: Exam 1 score of 45 and an Exam 2 score of 85
print(sigmoid(np.matmul(theta, np.array([1, 45, 85]))))

