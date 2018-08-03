"""
svm for 2D dataset -- Example 2
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
data = loadmat('ex6data3.mat')

# setup data
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
m = X.shape[0]
n = X.shape[1]

# visualize data
plt.figure(1)
for i in range(m):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red')
    elif y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='green')
plt.title('Example Dataset 3')
plt.xlabel('X1')
plt.ylabel('X2')


def gaussian_kernel(training_ex, landmark, sigma=0.1):
    """
    enter 2, 1D np arrays
    data provided: x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2; should get 0.324652
    """
    return np.exp(-(np.linalg.norm(training_ex - landmark) ** 2 / (2 * (sigma ** 2))))


def all_features(inputs, comparison, sigma=0.1):
    """
    returns matrix of similarity comparisons for ALL points in X
    """
    # to fill in f, array of features w shape (863, 863)
    f = np.zeros((inputs.shape[0], comparison.shape[0]))
    # iterate over every example twice to make every possible comparison
    for i in range(inputs.shape[0]):
        for j in range(comparison.shape[0]):
            f[i, j] = gaussian_kernel(inputs[i], comparison[j], sigma=sigma)
    return f


# find best values of sigma and C iterating thru all combinations of possible values
hyperparameter_ls = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
most_correct = 0
# iterate thru all combinations of possible values
for i in hyperparameter_ls:
    for j in hyperparameter_ls:
        # get all gaussian features for X and X_val, testing different sigma values
        X_features = all_features(X, X, sigma=i)
        Xval_features = all_features(Xval, X, sigma=i)

        # train SVM using X_features, testing different C values
        clf = svm.SVC(kernel='linear', C=j)
        clf.fit(X_features, y.ravel())

        # test on validation set
        correct = 0
        for k in range(len(Xval)):
            if clf.predict(np.array([Xval_features[k]])) == yval[k]:
                correct += 1

        # given values result in best performance, update most_correct and save sigma and C values
        if correct > most_correct:
            most_correct = correct
            best_sigma = i
            best_C = j

print('Best Accuracy: %0.2f' % (most_correct / Xval.shape[0]))
print('Best sigma:', best_sigma)
print('Best C:', best_C)


# to show boundary: train SVM with best hyperparameters
clf = svm.SVC(kernel='linear', C=best_C)
clf.fit(all_features(X, X, sigma=best_sigma), y.ravel())


def fill_features(input_point, sigma=0.1):
    """
    returns vector of similarity comparisons with X for a given value input_point, needed to plot predictions
    """
    # to fill in f, array of features w shape (863, 1)
    f = np.zeros(X.shape[0])
    # iterate over every example to compare
    for i in range(m):
        f[i] = gaussian_kernel(input_point, X[i], sigma=sigma)
    return f


# plot predictions
plt.figure(2)
X1_increments = np.linspace(-0.6, 0.4, 100)
X2_increments = np.linspace(-0.7, 0.6, 130)
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


