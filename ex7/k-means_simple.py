""" K-means Clustering on 2D dataset"""

##############################################################################################################
# IMPORT PACKAGES, LOAD AND VISUALIZE DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style; style.use('ggplot')

""" COULD ALSO LOAD ex7data1.mat """
from scipy.io import loadmat
data = loadmat('ex7data2.mat')
X = data['X']
m = X.shape[0]


def show_data():
    """ Plots initial data """
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], color='black')
    plt.title('Example Dataset 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


##############################################################################################################
# FUNCTIONS FOR CLUSTERING
##############################################################################################################
def random_initialization(k):
    """ Returns k randomly choosen points in X as initial centroids """
    centroid_ls = []
    for i in range(k):
        centroid_ls.append(X[np.random.randint(0, X.shape[0])])
    return np.asarray(centroid_ls)


def cluster_assigment(centroids):
    """ Finds closest centroid for every point in X, returns array (X.shape[0],) of indices of closest centroids """
    assignments = np.full(X.shape[0], np.nan)
    for i in range(m):
        min_error = np.inf
        for j in range(len(centroids)):
            instance_error = np.linalg.norm(X[i] - centroids[j]) ** 2
            if instance_error < min_error:
                min_error = instance_error
                assignments[i] = j
    return assignments


def move_centroid(centroids, assigments):
    """ Moves centroids to mean of their matched points, returns new locations """
    for i in range(len(centroids)):
        relevant_points = X[np.where(assigments == i)]
        centroids[i] = sum(relevant_points) / len(relevant_points)
    return centroids


def learn(k, iterations=100):
    """ Execute k-means algorithm """
    u = random_initialization(k)
    c = cluster_assigment(u)
    for i in range(iterations):
        c = cluster_assigment(u)
        u = move_centroid(u, c)
    return u, c


##############################################################################################################
# FUNCTIONS FOR VISUALIZATION
##############################################################################################################
def show_clusters(u, c, k):
    """ Show learned clusters, up to 10 """
    plt.clf()
    color_ls = ['skyblue', 'green', 'tomato', 'gold', 'indigo', 'khaki', 'pink', 'navy', 'orange', 'grey']
    for i in range(k):
        plt.scatter(X[np.where(c == i)][:, 0], X[np.where(c == i)][:, 1], color=color_ls[i])
    plt.scatter(u[:, 0], u[:, 1], s=100, marker='x', color='black')
    plt.title('Example Dataset 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    return None


def test_k(K):
    """ Display clustering for different k values"""
    u_, c_ = learn(K)
    show_clusters(u_, c_, K)
    return None


# run program, try different k values
test_k(3)


