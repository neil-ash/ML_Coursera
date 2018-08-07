""" SVM for email spam classification """

import numpy as np
from sklearn import svm
import re
from nltk.stem import PorterStemmer


##########################################################################################################
# LOAD DATA
##########################################################################################################
# needed to load data (as np arrays), puts data into numpy column vectors, shape == (length, 1)
from scipy.io import loadmat
spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')

# X and y for train and test
X = spam_train['X']
y = spam_train['y']
Xtest = spam_test['Xtest']
ytest = spam_test['ytest']


##########################################################################################################
# TRAIN SVM
##########################################################################################################
# train a linear svm
clf = svm.SVC(kernel='linear', probability=True, C=0.1)
clf.fit(X, y.ravel())

# test svm on test set
print('\nAccuracy on test set: %0.3f' % clf.score(Xtest, ytest), sep='')


##########################################################################################################
# SET UP FOR PREDICTING NEW EMAIL CLASSIFICATION
##########################################################################################################
# save learned parameters as theta, find index of largest postive weights, add 1 to get word indices
theta = clf.coef_[0].tolist()
top_idx = sorted(range(len(theta)), key=lambda i: theta[i])[-15:]
top_idx = [i + 1 for i in top_idx]
top_idx = top_idx[::-1]

# create dictionary after reading from vocab.txt file
vocab_file = open('vocab.txt', 'r')
vocab_dict = {}
for line in vocab_file:
    key, value = line.strip().split(None)
    vocab_dict[key.strip()] = value.strip()
vocab_file.close()

# print words with large, positive weights in order of largest -> smallest weight
print('\nTop 15 predictors of spam:', sep='')
for i in range(len(top_idx)):
    print('\t', i + 1, '. ', vocab_dict[str(top_idx[i])], sep='')


# SUPER IMPORTANT: switches keys and values in vocab_dict
vocab_dict = {v: k for k, v in vocab_dict.items()}


##########################################################################################################
# FUNCTIONS TO PREPROCESS AND MAKE PREDICTIONS ON NEW EMAILS
##########################################################################################################
def preprocess(filename):
    """
    returns sparse vector when given filename as string
    to use: preprocess('emailSample1.txt')
    """
    # reads in contents of email as a string
    file = open(filename, 'r')
    email_contents = file.read().replace('\n', '')
    file.close()
    # makes everything lowercase
    email_contents = email_contents.lower()
    # replace < > expressions with spaces
    email_contents = re.compile('<[^<>]+>').sub(' ', email_contents)
    # replace digit numbers with word number
    email_contents = re.compile('[0-9]+').sub(' number ', email_contents)
    # replace strings starting with http:// or https:// with httpaddr
    email_contents = re.compile('(http|https)://[^\s]*').sub(' httpaddr ', email_contents)
    # replace email addresses (strings with '@' in middle) with emailaddr
    email_contents = re.compile('[^\s]+@[^\s]+').sub(' emailaddr ', email_contents)
    # replace $ with dollar
    email_contents = re.compile('[$]+').sub(' dollar ', email_contents)
    # remove punctuation
    email_contents = re.split('[ @$/#.-:&*+=\[\]?!(){},''">_<;%\n\r]', email_contents)
    # remove empty strings
    email_contents = [word for word in email_contents if len(word) > 0]
    # replace words with their stems
    ps = PorterStemmer()
    final_word_ls = []
    word_indices = []
    for word in email_contents:
        # remove anything other than letters/numbers from word
        word = re.compile('[^a-zA-Z0-9]').sub('', word).strip()
        word = ps.stem(word)
        final_word_ls.append(word)
        # look up the word in the vocab_dict, if exists, then add index to word_indices
        if word in vocab_dict:
            word_indices.append(int(vocab_dict.get(word)))
    # now make sparse vectors for words
    vector = np.zeros(len(vocab_dict))
    for i in range(len(word_indices)):
        vector[word_indices[i]] = 1
    return vector


def predict(filename):
    """
    input filename with quotes, displays prediction statement
    """
    prediction = clf.predict(np.array([preprocess(filename)]))[0]
    if prediction == 1:
        print('\nProgram predicts SPAM!')
        print('(%0.3f confidence)\n' % clf.predict_proba((np.array([preprocess(filename)])))[0, 1], sep='')
    elif prediction == 0:
        print('\nProgram predicts NOT spam!\n')
        print('(%0.3f confidence)\n' % clf.predict_proba((np.array([preprocess(filename)])))[0, 0], sep='')
    return None


# to make a prediction:
# predict('mytests\\myspam1.txt')


