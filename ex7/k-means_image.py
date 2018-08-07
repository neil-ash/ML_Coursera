""" K-means Clustering for image compression """

##############################################################################################################
# IMPORT PACKAGES AND LOAD DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# load and display data
original_A = plt.imread('bird_small.png')
plt.figure(1)
plt.title('Original Image')
plt.imshow(original_A)

# reshape data to (128 ** 2, 3) w/ 3 as RGB colors
A = original_A.reshape(-1, 3)
m = A.shape[0]


##############################################################################################################
# FUNCTIONS FOR CLUSTERING
##############################################################################################################
def random_initialization(k):
    """ Returns k randomly choosen points in X as initial centroids """
    centroid_ls = []
    for i in range(k):
        centroid_ls.append(A[np.random.randint(0, A.shape[0])])
    return np.asarray(centroid_ls)


def cluster_assigment(centroids):
    """ Finds closest centroid for every point in X, returns array (X.shape[0],) of indices of closest centroids """
    assignments = np.full(A.shape[0], np.nan)
    for i in range(m):
        min_error = np.inf
        for j in range(len(centroids)):
            instance_error = np.linalg.norm((A[i] - centroids[j]).astype(np.float64)) ** 2
            if instance_error < min_error:
                min_error = instance_error
                assignments[i] = j
    return assignments.astype(int)


def move_centroid(centroids, assigments):
    """ Moves centroids to mean of their matched points, returns new locations """
    for i in range(len(centroids)):
        relevant_points = A[np.where(assigments == i)]
        centroids[i] = sum(relevant_points) / len(relevant_points)
    return centroids


def learn(k, iterations=25):
    """ Execute k-means algorithm """
    u = random_initialization(k)
    c = cluster_assigment(u)
    for i in range(iterations):
        c = cluster_assigment(u)
        u = move_centroid(u, c)
    return u, c


##############################################################################################################
# EXECUTE FUNCTIONS TO PERFORM CLUSTERING
##############################################################################################################
# get mapping of points in A onto 16 clusters C, described by U
vectors, indices = learn(16)

# reconstruct image after compression by reshaping
reconstructed = vectors[indices].reshape((original_A.shape[0], original_A.shape[1], original_A.shape[2]))

# show new image
plt.figure(2)
plt.title('Reconstructed Image')
plt.imshow(reconstructed)
plt.show()
