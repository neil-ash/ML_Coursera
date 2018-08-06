""" Implementation of a pre-trained neural network for MNIST dataset """

##########################################################################################################
# IMPORT PACKAGES AND PREPARE DATA
##########################################################################################################
import numpy as np

# needed to load data (as np arrays)
from scipy.io import loadmat
data = loadmat('ex3data1.mat')
weights = loadmat('ex3weights.mat')

# weights
theta1 = weights.get('Theta1')      # theta1: weights from 1 -> 2
theta2 = weights.get('Theta2')      # theta2: weights from 2 -> 3

# x and y
X = data['X']                       # x: 2-D array w 5000 training examples, each as a 400 element array
y = data['y'].ravel()               # y: 1-D array of 5000 labels

# m and n
m = X.shape[0]                      # number of training examples = 5000
n = X.shape[1]                      # number of features = 400


#########################################################################################################
# SETUP FEEDFORWARD NETWORK
##########################################################################################################
def sigmoid(z):
    """ Returns output of sigmoid for input z, works with np arrays element-wise """
    return 1 / (1 + np.exp(-z))


def predict(index):
    """ Make a prediction given an index in X (input) """
    inputs = np.concatenate((np.array([1]), X[index]))
    hidden_activations = sigmoid(np.matmul(theta1, inputs))
    hidden_activations = np.concatenate((np.array([1]), hidden_activations))
    outputs = sigmoid(np.matmul(theta2, hidden_activations))
    return outputs.argmax() + 1


#########################################################################################################
# EVALUATE NETWORK
##########################################################################################################
correct = 0
for i in range(m):
    if predict(i) == y[i]:
        correct += 1

print('\nTotal accuracy: %0.3f\n' % (correct / m), sep='')

