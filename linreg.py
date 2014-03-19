from __future__ import division
import numpy as np
import random


trainfile = open("SSFRTrain2014.dt", "r")
testfile = open("SSFRTest2014.dt", "r")

"""
This function reads in the files, strips by newline and splits by space char. 
It returns the labels as a 1D list and the features as one numpy array per row.
"""
def read_data(filename):
	features = np.array([])
	labels = np.array([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		features = np.append(features, l[:-1])
		labels = np.append(labels, l[-1])
	features = np.reshape(features, (-1, len(l)-1))
	return features, labels

X_train, y_train = read_data(trainfile)
X_test, y_test = read_data(testfile)
