""" Anomaly detection for a higher dimension dataset """

##############################################################################################################
# IMPORT PACKAGES AND LOAD DATA
##############################################################################################################
import numpy as np
from scipy.io import loadmat

# load data, make labels (yval) 1D
data = loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].ravel()
m = X.shape[0]
n = X.shape[1]


##############################################################################################################
# FUNCTIONS FOR ANOMALY DETECTION
##############################################################################################################
def means_and_variances(inputs):
    """ Return n-dimensionals vectors holding means and variances of inputs """
    means = []
    variances = []
    for i in range(inputs.shape[1]):
        means.append(np.sum(inputs[:, i]) / inputs.shape[0])
        variances.append(np.sum((inputs[:, i] - means[i]) ** 2) / inputs.shape[0])
    return np.asarray(means), np.asarray(variances)


def single_gaussian(inputs, mean, variance):
    """ Returns probability for a given input with given mean and variance """
    return (1 / (np.sqrt((2 * np.pi) * variance))) * np.exp(-(1 / 2) * (((inputs - mean) ** 2) / variance))


def multivariate_gaussian(inputs, means, variances):
    """ Returns probabilities for given inputs (m, n) with given means and variance in shape (m,)"""
    # start with list of all ones
    p = [1 for i in range(inputs.shape[0])]
    for i in range(inputs.shape[0]):
        # iterate over features
        for j in range(inputs.shape[1]):
            # compute probability p
            p[i] *= single_gaussian(inputs[i, j], means[j], variances[j])
    return np.asarray(p)


def select_threshold(means, variances):
    """ Returns the best value of epsilon, choosen by F1 score on validation set """
    # initialize things to optimize
    best_F1 = 0
    best_epsilon = None
    # find p for validation set (single value can be used for rest of function)
    p = multivariate_gaussian(Xval, means, variances)
    # iterate over possible epsilon values (min(p) -> max(p) for 1000 steps), finding F1 each time
    possible_epsilons = np.arange(np.min(p), np.max(p), (np.max(p) - np.min(p)) / 1000)
    for epsilon in possible_epsilons:
        # if p < epsilon, then hypothesis is 1, otherwise hypothesis is 0
        hypothesis = np.asarray([1 if i < epsilon else 0 for i in p])
        # count up tp, fp, fn
        tp = np.sum(np.logical_and(hypothesis == 1,  yval == 1).astype(float))
        fp = np.sum(np.logical_and(hypothesis == 1,  yval == 0).astype(float))
        fn = np.sum(np.logical_and(hypothesis == 0,  yval == 1).astype(float))
        # find precision and recall
        if (tp + fp) != 0 and (tp + fn) != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0
            recall = 0
        # find F1 score
        F1 = ((2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0)
        # if best results, save results and epsilon
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    return best_epsilon


def find_amonalies(means, variances, epsilon):
    """ Returns anomalies in training data """
    p = multivariate_gaussian(X, means, variances)
    hypothesis = []
    for i in p:
        hypothesis.append(1 if i < epsilon else 0)
    hypothesis = np.asarray(hypothesis)
    return X[np.where(hypothesis == 1)]


##############################################################################################################
# EXECUTE FUNCTIONS TO FIND ANOMALIES
##############################################################################################################
# set parameters
mean_, variance_ = means_and_variances(X)
threshold = select_threshold(mean_, variance_)

# find number of anomalies in training set
print('\n%d anomalies in training set.\n' % find_amonalies(mean_, variance_, threshold).shape[0])

