"""
svm for 2D dataset -- Example 2
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
data = loadmat('ex6data2.mat')

# setup data
X = data['X']
y = data['y']
m = X.shape[0]
n = X.shape[1]

# visualize data
plt.figure(1)
for i in range(m):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red')
    elif y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='green')
plt.title('Example Dataset 2')
plt.xlabel('X1')
plt.ylabel('X2')


def gaussian_kernel(training_ex, landmark, sigma=0.1):
    """
    enter 2, 1D np arrays
    data provided: x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2; should get 0.324652
    """
    return np.exp(-(np.linalg.norm(training_ex - landmark) ** 2 / (2 * (sigma ** 2))))


def all_features(sigma=0.1):
    """
    returns matrix of similarity comparisons for ALL points in X
    """
    # to fill in f, array of features w shape (863, 863)
    f = np.zeros((X.shape[0], X.shape[0]))
    # iterate over every example twice to make every possible comparison
    for i in range(m):
        for j in range(m):
            f[i, j] = gaussian_kernel(X[i], X[j], sigma=sigma)
    return f


def fill_features(input_point, sigma=0.1):
    """
    returns vector of similarity comparisons with X for a given value input_point
    """
    # to fill in f, array of features w shape (863, 1)
    f = np.zeros(X.shape[0])
    # iterate over every example to compare
    for i in range(m):
        f[i] = gaussian_kernel(input_point, X[i], sigma=sigma)
    return f


# get all features
gaussian_features = all_features()

# train SVM using gaussian features
clf = svm.SVC(kernel='linear', C=1)
clf.fit(gaussian_features, y.ravel())

# plot predictions
plt.figure(2)
X1_increments = np.linspace(0, 1, 100)
X2_increments = np.linspace(0.4, 1, 60)
for i in X1_increments:
    for j in X2_increments:
        prediction = clf.predict(np.array([fill_features(np.array([i, j]))]))
        if prediction == 0:
            plt.scatter(i, j, color='red')
        elif prediction == 1:
            plt.scatter(i, j, color='green')
plt.title('Predictions')
plt.xlabel('X1')
plt.ylabel('X2')

# show both plots: actual data and predictions
plt.show()


""" to see different decision boundaries, try with C = 100, C = 0.001 """

