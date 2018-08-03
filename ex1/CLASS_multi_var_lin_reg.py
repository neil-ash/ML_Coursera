"""
implementation of multivariable linear regression as a class
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


########################################################################################################################
# SPICY HOT CLASS
########################################################################################################################
class MultivariableLinearRegression:
    def __init__(self):
        # load data
        data = np.genfromtxt('ex1data2.txt', delimiter=',')
        # save data as class variables
        self.x1 = data[:, 0]   # size of house
        self.x2 = data[:, 1]   # num rooms
        self.y = data[:, 2]    # price
        self.m = self.y.size   # number of points
        # matrix to hold all x values (including x0 = 1)
        self.x_all = np.vstack((np.ones(self.m), self.x1, self.x2))
        # hold values of means and stds for features and labels
        self.mean_x1 = self.mean_x2 = self.mean_y = None
        self.std_x1 = self.std_x2 = self.std_y = None
        # theta: parameters, initialize to 0
        self.theta = np.zeros(3)

    @staticmethod
    def normalize(feature):
        """
        input a feature as an array, returns normalized feature, feature mean, feature std
        """
        # subtract mean, saving mean
        mean = sum(feature) / feature.size
        feature -= mean
        # divide by std, saving std
        std = np.std(feature)
        feature /= std
        return feature, mean, std

    def normalize_features(self):
        """
        actually normalizes features
        """
        self.x1, self.mean_x1, self.std_x1 = self.normalize(self.x1)
        self.x2, self.mean_x2, self.std_x2 = self.normalize(self.x2)
        self.y, self.mean_y, self.std_y = self.normalize(self.y)
        # important to update x_all
        self.x_all = np.vstack((np.ones(self.m), self.x1, self.x2))
        return None

    @staticmethod
    def hypothesis(theta, inputs):
        """
        input parameters theta and inputs x1, x2 ..., outputs hypothesis result
        """
        return np.matmul(theta, inputs)

    def compute_cost(self):
        """
        computes cost for given parameters theta with all points
        """
        return (1 / (2 * self.m)) * sum((self.hypothesis(self.theta, self.x_all) - self.y) ** 2)

    def gradient_descent(self, iterations=1500, alpha=0.01):
        """
        performs gradient descent
        """
        # set up cost graph
        plt.figure(1).clf()
        ax1 = plt.figure(1).add_subplot(111)
        ax1.set_title('Cost vs Iterations')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost')
        # adjust theta every iteration
        for i in range(iterations):
            # plot cost for a given iteration
            ax1.scatter(i, self.compute_cost(), color='red')
            # find delta: gradient of cost function
            delta = (1 / self.m) * np.matmul((self.hypothesis(self.theta, self.x_all) - self.y), self.x_all.T)
            # adjust theta in the direction of the negative gradient w/ step proportional to learning rate
            self.theta -= alpha * delta
        # print how successful model was and show cost graph
        print('Trend line equation (normalized): y = %.2f * x2 + %.2f * x1 + %.2f' % (self.theta[2], self.theta[1], self.theta[0]), sep='')
        print('Final error: %.2f' % self.compute_cost())
        plt.show()
        return self.theta

    def show_plots(self):
        """
        to display scatter plot and trend line
        """
        ax2 = plt.figure(2).add_subplot(111, projection='3d')
        ax2.set_title('Predicting Housing Prices')
        ax2.set_xlabel('House Size')
        ax2.set_ylabel('Number of Rooms')
        ax2.set_zlabel('Price')
        # scatter plot of data
        ax2.scatter(self.x1, self.x2, self.y, color='pink')
        # 2 3D points needed to plot trend line
        x1_show = [min(self.x1), max(self.x1)]
        x2_show = [min(self.x2), max(self.x2)]
        y_show = [np.matmul(self.theta, [1, x1_show[0], x2_show[0]]), np.matmul(self.theta, [1, x1_show[1], x2_show[1]])]
        # plot trend line
        ax2.plot(x1_show, x2_show, y_show)
        # actually show graph
        plt.show()
        return None

    def predict(self, ex1, ex2):
        """
        enter a house size and number of rooms in order to predict price
        """
        # not yet normalized inputs
        print('For x1 = %.0f and x2 = %.0f:' % (ex1, ex2))
        # normalized inpurs
        ex1 = (ex1 - self.mean_x1) / self.std_x1
        ex2 = (ex2 - self.mean_x2) / self.std_x2
        print('\tNormalized to x1 := %.2f and x2 := %.2f' % (ex1, ex2))
        # make a prediction
        unadjusted_prediction = self.hypothesis(self.theta, [1, ex1, ex2])
        print('\tNormalized prediction: %.2f' % unadjusted_prediction)
        # normalize prediction to get a result that makes sense
        adjusted_prediction = unadjusted_prediction * self.std_y + self.mean_y
        print('\tTrue prediction: %.0f' % adjusted_prediction)
        return adjusted_prediction

    def normal_eq(self):
        """
        returns parameters theta as calculated by the normal equation
        """
        # save temporary variables
        x_all = self.x_all
        y = self.y
        # reformat matrix/vector to work with normal eq
        x_all = x_all.T
        y = np.array([y])
        y = y.T
        # normal eq
        theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(x_all.T, x_all)), x_all.T), y)
        print('Trend line equation (exact):      y = %.0f * x2 + %.0f * x1 + %.0f' % (theta[2], theta[1], theta[0]), sep='')
        return theta

    def run(self):
        """
        steps to do linear regression
        """
        self.theta = np.zeros(3)
        self.normal_eq()
        self.normalize_features()
        self.gradient_descent()
        self.show_plots()
        self.predict(1650, 3)
        return None


########################################################################################################################
# TEST INSTANCE OF THE CLASS
########################################################################################################################
print('\n', end='')
test = MultivariableLinearRegression()
test.run()
print('\n', end='')
