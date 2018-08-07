""" Anomaly Detection on 2D dataset """

##############################################################################################################
# IMPORT PACKAGES, LOAD AND VISUALIZE DATA
##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import style; style.use('seaborn')
from scipy.io import loadmat

# load data
data = loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']
m = X.shape[0]
n = X.shape[1]

# setup axes for multiple plots on same grid
ax = plt.figure(1).add_subplot(111)


def show_data():
    """ Visualize unlabeled data """
    ax.scatter(X[:, 0], X[:, 1], color='skyblue')
    ax.set_title('Server Data')
    ax.set_xlabel('Latency')
    ax.set_ylabel('Throughput')
    return None


##############################################################################################################
# FUNCTIONS FOR ANOMALY DETECTION
##############################################################################################################
def estimate_gaussian(inputs):
    """ Return n-dimensionals vectors holding means and variances of inputs """
    means = []
    variances = []
    for i in range(inputs.shape[1]):
        means.append(np.sum(inputs[:, i]) / inputs.shape[0])
        variances.append(np.sum((inputs[:, i] - means[i]) ** 2) / inputs.shape[0])
    return np.asarray(means), np.asarray(variances)


def show_contours(means, variances):
    """ Visualize countours """
    show_data()
    for i in range(2, 8):
        ellipse = Ellipse((means[0], means[1]),
                          width=i * 2 * variances[0],
                          height=i * 2 * variances[1],
                          angle=0, fill=False, color='black')
        ax.add_artist(ellipse)
    return None


def select_threshold(means, variances):
    """ Returns the best value of epsilon, choosen by F1 score """
    best_F1 = 0
    best_epsilon = None
    # iterate over possible epsilon values
    for epsilon in np.arange(0.001, 0, -0.00001):
        # initialize counters to 0
        tp = 0; fp = 0; fn = 0
        # iterate over examples in validation set
        for i in range(Xval.shape[0]):
            # set p = 1, since use p *= in next loop
            p = 1
            # iterate over features
            for j in range(Xval.shape[1]):
                # compute probability p
                p *= (1 / ((np.sqrt(2 * np.pi)) * np.sqrt(variances[j]))) * \
                     np.exp(-((Xval[i, j] - means[j]) ** 2) / (2 * variances[j]))
            # done with iterations for a single example: if p < epsilon, then hypothesis = 1, otherwise hypothesis = 0
            hypothesis = (1 if p < epsilon else 0)
            # add to counters of tp, fn, fp
            if yval[i] == 1 and hypothesis == 1:   tp += 1
            elif yval[i] == 1 and hypothesis == 0: fn += 1
            elif yval[i] == 0 and hypothesis == 1: fp += 1
        # done with iterations over all examples, now find precision and recall
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
    """ Returns anomalies in training data and plots them in red """
    anomalies = []
    # iterate over training examples
    for i in range(X.shape[0]):
        # set p = 1, since use p *= in next loop
        p = 1
        # iterate over features
        for j in range(X.shape[1]):
            # compute probability p
            p *= (1 / ((np.sqrt(2 * np.pi)) * np.sqrt(variances[j]))) * \
                 np.exp(-((X[i, j] - means[j]) ** 2) / (2 * variances[j]))
        # after iterating thru features, can check if p below threshold
        if p < epsilon:
            ax.scatter(X[i, 0], X[i, 1], marker='x', color='red')
            anomalies.append(X[i])
    return np.asarray(anomalies)


##############################################################################################################
# EXECUTE FUNCTIONS TO FIND ANOMALIES
##############################################################################################################
# set parameters
mean, variance = estimate_gaussian(X)
threshold = select_threshold(mean, variance)

# create visualizations
show_contours(mean, variance)
find_amonalies(mean, variance, threshold)


