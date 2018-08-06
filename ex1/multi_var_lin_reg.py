""" Linear regression with mulitple variables """

##########################################################################################################
# IMPORT PACKAGES AND DATA
##########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('seaborn')
from mpl_toolkits.mplot3d import Axes3D

# load data into np arrays for X, y
data = np.genfromtxt('ex1data2.txt', delimiter=',')
X1 = data[:, 0]                             # size of house
X2 = data[:, 1]                             # num rooms
y = data[:, 2]                              # price
m = y.size                                  # number of points
X = np.vstack((np.ones(m), X1, X2))     # X1 and X2

# original values to use later
X1_original = X1.copy()
X2_original = X2.copy()

# initialize parameters as 3-element array of 0s
initial_theta = np.zeros(3)


##########################################################################################################
# FUNCTIONS FOR SETUP AND TRAINING
##########################################################################################################
def normalize_features(X1_, X2_):
    """ Function to normalize features -- center at 0, scale by std """
    # subtract mean
    X1_ -= sum(X1_) / X1_.size
    X2_ -= sum(X2_) / X2_.size
    # scale by std
    X1_ /= np.std(X1_)
    X2_ /= np.std(X2_)
    return X1_, X2_


def hypothesis(theta, inputs=X):
    """ For all values in X, computes hypothesis h(x) = theta[0] + theta[1] * X """
    return np.matmul(theta, inputs)


def compute_cost(theta):
    """ Computes value of cost function (MSE) for given parameters theta """
    return (1 / (2 * m)) * sum((hypothesis(theta) - y) ** 2)


def gradient_descent(theta, iterations=1500, alpha=0.01):
    """ Performs gradient descent """
    # setup cost graph
    plt.figure(1).clf()
    ax1 = plt.figure(1).add_subplot(111)
    ax1.set_title('Cost vs Iterations')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')

    # adjust theta every iteration
    for i in range(iterations):
        ax1.scatter(i, compute_cost(theta), color='red')
        # find delta: gradient of cost function
        delta = (1 / m) * np.matmul((hypothesis(theta) - y), X.T)
        # adjust theta in the direction of the negative gradient w/ step proportional to learning rate
        theta = theta - alpha * delta

    # print how successful model was
    print('\nFinal error: %d' % compute_cost(theta))
    return theta


##########################################################################################################
# EXECUTE FUNCTIONS AND SHOW RESULTS
##########################################################################################################
# normalize features
X1, X2 = normalize_features(X1, X2)
X[1:, :] = X1, X2

# actually run gradient descent
final_theta = gradient_descent(initial_theta)
print('Best values of theta: %d, %d, %d' % (final_theta[0], final_theta[1], final_theta[2]))

# plot results in 3D -- may not be best way to visualize...
ax2 = plt.figure(2).add_subplot(111, projection='3d')
ax2.set_title('Price vs Size, Number of Rooms')
ax2.set_xlabel('House Size')
ax2.set_ylabel('Number of Rooms')
ax2.set_zlabel('Price')
# scatter plot of data
ax2.scatter(X1, X2, y, color='pink')
# 2 3D points needed to plot trend line
x1_show = [min(X1), max(X1)]
x2_show = [min(X2), max(X2)]
y_show = [np.matmul(final_theta, [1, x1_show[0], x2_show[0]]), np.matmul(final_theta, [1, x1_show[1], x2_show[1]])]
# plot trend line
ax2.plot(x1_show, x2_show, y_show)

# to all graphs
plt.show()

# to make a new prediction, first normalize inputs
input1 = 1650 - sum(X1_original) / X1_original.size
input2 = 3 - sum(X2_original) / X2_original.size
input1 /= np.std(X1_original)
input2 /= np.std(X2_original)
# actually make prediction
print('Price of a house with size 1650, 3 rooms: %d\n'
      % hypothesis(final_theta, inputs=np.array([1, input1, input2])))
