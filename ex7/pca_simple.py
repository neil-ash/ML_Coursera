""" Principal Component Analysis on 2D dataset """

"""
Note) Could've done procedurally, no need for functions
"""

##############################################################################################################
# IMPORT PACKAGES, LOAD AND VISUALIZE DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

from scipy.io import loadmat
data = loadmat('ex7data1.mat')
X = data['X']
m = X.shape[0]


def show_data(recovered=True):
    """ Plots initial data """
    if recovered: plt.scatter(X_recovered[:, 0], X_recovered[:, 1], color='tomato')
    plt.scatter(X[:, 0], X[:, 1], color='skyblue')
    plt.title('Example Dataset 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


##############################################################################################################
# PCA FUNCTIONS
##############################################################################################################
def normalize(inputs):
    """ Subtract mean and divide by std, call as X[0] = normalize(X[:, 0]); X[1] = normalize(X[:, 1])"""
    inputs = inputs.copy()
    inputs -= np.mean(inputs)
    inputs /= np.std(inputs)
    return inputs


def svd(inputs):
    """ Creates covariance matrix and returns eigenvectors (and more) """
    sigma = (1 / inputs.shape[0]) * np.matmul(inputs.T, inputs)
    U_, S_, V_ = np.linalg.svd(sigma)
    return U_, S_, V_


def projection(inputs, U_, k):
    """ Projects inputs onto U_ with dimenion k """
    U_ = U_[:, :k]
    return np.dot(inputs, U_)


def recover(inputs, U_, k):
    """ Recover approximation of original data given projected data inputs, projection vector U_, and dimension k"""
    U_ = U_[:, :k]
    return np.matmul(U_, inputs.T).T


##############################################################################################################
# EXECUTE FUNCTIONS TO PERFORM PCA
##############################################################################################################
# normalize features
X[:, 0] = normalize(X[:, 0])
X[:, 1] = normalize(X[:, 1])

# find eigenvectors of covariance matrix
U, S, V = svd(X)
print('\nEigenvectors of covariance matrix: %0.3f, %0.3f' % (U[:, :1][0], U[:, :1][0]))
print('(points will be projected onto eigenvectors)\n')

# project X onto eigenvectors, then recover X
X_projection = projection(X, U, 1)
X_recovered = recover(X_projection, U, 1)

# show data and projections
show_data()


