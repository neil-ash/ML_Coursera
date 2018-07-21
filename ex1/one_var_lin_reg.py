"""
ex1: linear regression with one variable
"""

##########################################################################################################
# IMPORT PACKAGES AND VISUALIZE DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# load data into np arrays for X, y
data = np.genfromtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

# plot points
ax1 = plt.figure(1).add_subplot(111)
plt.scatter(X, y, color='pink')
plt.title('Profit vs Population')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')


##########################################################################################################
# PREPARE FOR TRAINING
##########################################################################################################
# add column of ones to X (for X0 = 1)
X = np.vstack((np.ones(m), X))

# initialize parameters theta[0] (y-int), theta[1] (slope)
theta = np.zeros(2)

# number of iterations, learning rate
iterations = 1500
alpha = 0.01


def hypothesis():
    """
    for all values in X, computes hypothesis h(x) = theta[0] + theta[1] * X
    """
    return np.matmul(theta, X)


def compute_cost():
    """
    computes value of cost function (MSE) for given parameters theta
    """
    return (1 / (2 * m)) * sum((hypothesis() - y) ** 2)


##########################################################################################################
# GRADIENT DESCENT
##########################################################################################################
# setup cost graph
ax2 = plt.figure(2).add_subplot(111)

# adjust theta every iteration
for i in range(iterations):
    # plot error, throwing away first two high values
    if i > 2:
        ax2.scatter(i, compute_cost(), color='red')
    # find delta: gradient of cost function
    delta = (1 / m) * np.matmul((hypothesis() - y), X.T)
    # adjust theta in the direction of the negative gradient w/ step proportional to learning rate
    theta = theta - alpha * delta


##########################################################################################################
# SHOW RESULTS
##########################################################################################################
# make cost function graph look nice
ax2.set_title('Cost vs Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')

# plot trend line
x_show = [min(X[1]), max(X[1])]
y_show = [np.matmul(theta, [1, min(X[1])]), np.matmul(theta, [1, max(X[1])])]
ax1.plot(x_show, y_show)

# show all graphs
plt.show()

# output equation
print('Trend line in form: y = %.2f * x + %.2f' % (theta[1], theta[0]))

# output predictions
print('A city with 35,000 people has an expected profit of $%.0f' % np.matmul([1, 35000], theta), sep='')
print('A city with 70,000 people has an expected profit of $%.0f' % np.matmul([1, 70000], theta), sep='')

