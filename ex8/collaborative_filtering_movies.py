""" Collaborative Filtering for movie reccomendations """

"""
Note) cost and gradient functions do not include R_ in regularization
"""

##############################################################################################################
# IMPORT PACKAGES, LOAD AND VISUALIZE DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# load data
data_ratings = loadmat('ex8_movies.mat')
data_parameters = loadmat('ex8_movieParams.mat')
Y = data_ratings['Y']                            # rating of movie i by user j (index (i, j)), (num_movies, num_users)
R = data_ratings['R']                            # if a movie i has been rated by user j (index (i, j)), shape of Y
X = data_parameters['X']                         # features (10) for each movie (num_movies, num_features)
Theta = data_parameters['Theta']                 # parameters for each user (num_users, num_features)
num_users = data_parameters['num_users'][0, 0]
num_movies = data_parameters['num_movies'][0, 0]
num_features = data_parameters['num_features'][0, 0]

# use subset of data to run faster for tests (setup from .m file)
subset_users = 4; subset_movies = 5; subset_features = 3
X_ = X[:subset_movies, :subset_features]
Theta_ = Theta[:subset_users, :subset_features]
Y_ = Y[:subset_movies, :subset_users]
R_ = R[:subset_movies, :subset_users]

# average rating for first movie (Toy Story)
print('Average rating for Toy Story: %.2f' % (np.sum(Y[0]) / np.count_nonzero(Y[0])))


def visualize_ratings():
    """ Shows plot of movie ratings """
    plt.figure(figsize=(6, 6 * (1682 / 943)))
    plt.imshow(Y, cmap='inferno')
    plt.colorbar()
    plt.ylabel('Movies (%d)' % num_movies, fontsize=20)
    plt.xlabel('Users (%d)' % num_users, fontsize=20)
    plt.title('Movie Ratings', fontsize=25)
    plt.show()


##############################################################################################################
# FUNCTIONS FOR COLLABORATIVE FILTERING
##############################################################################################################
def cost(x=X_, theta=Theta_, y=Y_, r=R_, LAMBDA=1.5):
    """ Returns unregularized total cost for a subset of the dataset """
    unregularized_cost = (1/2) * np.sum((r * (np.matmul(x, theta.T) - y)) ** 2)
    regularization_term = (LAMBDA/2) * (np.sum(theta ** 2) + np.sum(x ** 2))
    return unregularized_cost + regularization_term


def gradient(x=X_, theta=Theta_, y=Y_, r=R_, LAMBDA=1.5):
    """ Returns unregularized gradients for subsets of X and Theta """
    X_grad = np.matmul((r * np.matmul(x, theta.T) - y), theta) + LAMBDA * np.sum(x)
    Theta_grad = np.matmul((r * np.matmul(x, theta.T) - y).T, x) + LAMBDA * np.sum(theta)
    return X_grad, Theta_grad


def gradient_check(index, use):
    """ Verifies gradient numerically given index and use = 'X_' or use = 'Theta_' """
    if use == 'X_':
        epsilon = np.zeros(X_.shape)
        epsilon[index] = 0.000001
        return (cost(x=(X_ + epsilon)) - cost(x=(X_ - epsilon))) / (2 * epsilon[index])
    elif use == 'Theta_':
        epsilon = np.zeros(Theta_.shape)
        epsilon[index] = 0.000001
        return (cost(theta=(Theta_ + epsilon)) - cost(theta=(Theta_ - epsilon))) / (2 * epsilon[index])


def gradient_descent(x, theta, y, r, LAMBDA=0, learning_rate=0.000001, iterations=1000):
    """ Performs gradient descent to train model, returns optimized x and theta arrays """
    for i in range(iterations):
        X_grad, Theta_grad = gradient(x=x, theta=theta, y=y, r=r, LAMBDA=LAMBDA)
        x -= learning_rate * X_grad
        theta -= learning_rate * Theta_grad
        print(cost(x=x, theta=theta, y=y, r=r, LAMBDA=LAMBDA))
    return x, theta


def normalize(y):
    """ Perform mean normalization on ratings y, return mean-normalized y and mean of every movie in array """
    # for every movie, find mean rating
    y_mean = (np.sum(y, axis=1) / np.count_nonzero(y, axis=1)).reshape((-1, 1))
    y_norm = y - y_mean
    return y_norm, y_mean


##############################################################################################################
# MAKE PREDICTIONS
##############################################################################################################
# normalize Y and randomly initialize X and Theta
Y_normalized, Y_means = normalize(Y)
X = np.random.rand(num_movies, num_features) * 2 - 1
Theta = np.random.rand(num_users, num_features) * 2 - 1

# train using gradient descent
X_final, Theta_final = gradient_descent(x=X, theta=Theta, y=Y_normalized, r=R)


def predict(movie, user):
    """ Make a prediction given a movie index and user index """
    predictions = np.matmul(X_final, Theta_final.T) + Y_means
    return predictions[movie, user]
