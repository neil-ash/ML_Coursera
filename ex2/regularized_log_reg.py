""" Regularized logistic regression using gradient descent """

##########################################################################################################
# IMPORT PACKAGES AND PREPARE DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use('ggplot')

# get data
data = np.genfromtxt('ex2data2.txt', delimiter=',')
data = data.T

# plot data points: green = accepted, red = denied
for i in range(data[0].size):
    if data[2][i] == 0:
        plt.scatter(data[0][i], data[1][i], color='red')
    elif data[2][i] == 1:
        plt.scatter(data[0][i], data[1][i], color='green')
plt.title('Microchip Quality')
plt.xlabel('Quality Test #1')
plt.ylabel('Quality Test #2')

# set up data nicely
X1 = data[0]                                    # test 1
X2 = data[1]                                    # test 2
X = np.array([np.ones(X1.size), X1, X2])        # includes 1s, x1, x2
y = data[2]                                     # accepted (1) or rejected (0)
m = data[0].size                                # number of training examples
n = 28                                          # number of features (after running map_features)


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def map_features(input1, input2):
    """ Creates list w/ 28 arrays corresponding to 1, x1, x2...x1x2^5, x2^6 """
    degree = 6
    x_list = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            x_list.append((input1 ** (i - j)) * (input2 ** j))
    # return mapped features as an np array, include first entry of 1s
    if input1.size != 1:
        return np.concatenate((np.array([np.ones(input1.size)]), np.asarray(x_list)))
    else:
        return np.concatenate((np.ones(input1.size), np.asarray(x_list)))


def sigmoid(z):
    """ Returns sigmoid of input, works with np arrays (vectors and matrices) element-wise, same dimension as input """
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, input1, input2):
    """ Returns hypothesis for given parameters theta (28 element array), inputs (28 rows, any columns) """
    return sigmoid(np.matmul(theta, map_features(input1, input2)))


def cost(theta, L=1):
    """ Returns cost function for given parameters theta including regularization term """
    # start with unregularized
    unregularized_cost = (1 / m) * sum((-y) * np.log(hypothesis(theta, X1, X2)) -
                                       (1 - y) * np.log(1 - hypothesis(theta, X1, X2)))
    # include regularization term for all parameters EXCEPT theta[0]
    regularization_term = 0
    for i in range(1, n):
        regularization_term += theta[i] ** 2
    # scale by lambda
    regularization_term *= (L / (2 * m))
    # return full cost
    return unregularized_cost + regularization_term


def gradient(theta, L=1):
    """ Returns gradient (28 element array) of cost function for given parameters theta """
    # equation for gradient: use for all theta
    theta_gradient = (1 / m) * np.matmul((hypothesis(theta, X1, X2) - y), map_features(X1, X2).T)
    # inclusion of regularization term: for all theta EXCEPT theta0
    for i in range(1, n):
        theta_gradient[i] += (L / m) * theta[i]
    return theta_gradient


def gradient_descent(theta, learning_rate=0.03, iterations=10000):
    """ Performs gradient descent, returns optimal values of theta """
    for i in range(iterations):
        theta -= learning_rate * gradient(theta)
    return theta


##########################################################################################################
# SHOW OPTIMIZED THETA AND ACCURACY
##########################################################################################################
initial_theta = np.zeros(n)
final_theta = gradient_descent(initial_theta)

# plot decision boundary
a = np.linspace(-1, 1.5, 50)
b = np.linspace(-1, 1.5, 50)
c = np.zeros((len(a), len(b)))
for i in range(len(a)):
    for j in range(len(b)):
        c[i][j] = hypothesis(final_theta, a[i], b[j])
plt.contour(a, b, c, levels=[0.5], colors='black')
plt.show()

# compute accuracy
num_correct = 0
num_wrong = 0
for i in range(m):
    if hypothesis(final_theta, data[0, i], data[1, i]) >= 0.5 and data[2, i] == 1:
        num_correct += 1
    elif hypothesis(final_theta, data[0, i], data[1, i]) < 0.5 and data[2, i] == 0:
        num_correct += 1
    else:
        num_wrong += 1

# print stats
print('\nOptimized values of theta:', sep='')
for i in final_theta:
    print('%0.2f' % i)
print('\nAccuracy on training set: %0.2f' % (num_correct / m), '\n\n', sep='', end='')
