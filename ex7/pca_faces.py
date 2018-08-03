"""
principal component analysis on faces dataset
"""

##############################################################################################################
# IMPORT PACKAGES AND LOAD DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

# load data
from scipy.io import loadmat
data = loadmat('ex7faces.mat')
X = data['X']
m = X.shape[0]


##############################################################################################################
# PERFORM PCA
##############################################################################################################
# set k: dimension to reduce to (from n = 1024 -> k)
k = 100

# mean normalization (subtract mean from each feature so new mean is 0)
for i in range(X.shape[1]):
    X[:, i] -= np.mean(X[:, i])

# create principal components U (that minimize projection error) from covariance matrix sigma
sigma = (1 / X.shape[0]) * np.matmul(X.T, X)
U, S, V = np.linalg.svd(sigma)
principal_components = U[:, :k].T

# creates projection: project X onto principal components
X_projection = np.dot(X, principal_components.T)

# approximate X from reduced dimension
X_approximation = np.dot(X_projection, principal_components)


##############################################################################################################
# VIEW RESULTS
##############################################################################################################
def show_face(index, use=X):
    """
    Displays creepy looking face given its index and dataset:
    - X == original data
    - X_approximation == reconstruction after reduction
    - U == principal components for ALL data
    """
    plt.clf()
    plt.title('Face Example')
    plt.axis('off')
    # face needs to be flipped and rotated (like all .mat files of this type)
    plt.imshow(np.rot90(np.flip(np.reshape(use[index, :], (32, 32)), axis=1)), cmap=plt.cm.binary)
    plt.show()
    return None

