""" SVM for 2D, linearly separable dataset using sklearn """

from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

##########################################################################################################
# LOAD AND VISUALIZE DATA
##########################################################################################################
# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
data = loadmat('ex6data1.mat')

# setup data
X = data['X']
y = data['y']
m = X.shape[0]
n = X.shape[1]

# visualize data
for i in range(m):
    if y[i] == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red')
    elif y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='green')
plt.title('Example Dataset 1')
plt.xlabel('X1')
plt.ylabel('X2')


##########################################################################################################
# TRAIN SVM AND PLOT DECISION BOUNDARY
##########################################################################################################
# train an SVM using a linear kernel
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y.ravel())

# decision boundary eq in form: intercept + x1 * coeffficient[0] + x2 * coefficient[1] = 0
theta0 = clf.intercept_[0]
theta1 = clf.coef_[0][0]
theta2 = clf.coef_[0][1]
# points to plot: x-coors then y-coors
ptX1 = min(X[:, 0]), max(X[:, 0])
ptX2 = (-theta0 - theta1 * ptX1[0]) / theta2, (-theta0 - theta1 * ptX1[1]) / theta2
# actually plot
plt.plot(ptX1, ptX2, color='black')
plt.show()

