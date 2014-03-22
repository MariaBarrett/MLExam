from __future__ import division
import numpy as np
import random
from sklearn.svm import libsvm
from operator import itemgetter
from collections import Counter


trainfile = open("SGTrain2014.dt", "r")
testfile = open("SGTest2014.dt", "r")

"""
This function reads in the files, strips by newline and splits by space char. 
It returns the labels as a 1D list and the features as one numpy array per row.
"""
def read_data(filename):
	features = np.array([])
	labels = np.array([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(','),dtype='float')
		features = np.append(features, l[:-1])
		labels = np.append(labels, l[-1])
	features = np.reshape(features, (-1, len(l)-1))
	return features, labels



##############################################################################
#
#                      Normalizing and transforming
#
##############################################################################
"""
This function takes a dataset with class labels and computes the mean and the variance of each input feature 
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	mean = sum(data) / len(data)
	variance = sum((data - mean)**2) / len(data)
	return mean, variance
	

"""
This function expects a dataset.
It calls mean_variance to get the mean and the variance for each feature.
Then each datapoint is normalized using means and variance. 
"""
def meanfree(data):
	mean, variance = mean_variance(data)
	meanfree = (data - mean) / np.sqrt(variance)
	return meanfree

"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set and call a function to get mean and variance of the train set.
It uses these to transform all datapoints of the test set. 
It returns the transformed test set. 

"""
def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	transformed = (testset - meantrain) / np.sqrt(variancetrain)
	return transformed


def euclidean(ex1,ex2):
	"""
	This function takes two datapoints and calculates the euclidean distance between them.
	It expects two data points without class label 
	"""
	assert len(ex1) == len(ex2)
	inner = 0
	for i in xrange(len(ex1)):
		inner += (ex1[i] - ex2[i])**2 
	distance = np.sqrt(inner)
	return distance

def Jaakkola(X, y):
	G = []
	for i,n in enumerate(y):
		temp = []
		for j,k in enumerate(y):
			if n != k: #if not the same class:
				eu = euclidean(X[i], X[j])
				temp.append(eu) #temp is a list of eucidean distances
		temp = sorted(temp)
		G.append(temp[0]) #appending smallest euclidean distance
	sigma = np.median(np.array(G))
	return sigma

def fromsigmatogamma(sigma):
	gamma = 1 / (2*(sigma**2))
	return gamma

"""
This function splits the shuffled train set in s equal sized splits. 
It expects the features, the labels and number of slices. 
It starts by making a copy of the labels and features and shuffles them. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints belonging to s.
"""
def sfold(features, labels, s):
	featurefold = np.copy(features)
	labelfold = np.copy(labels)

	#random.shuffle(featurefold, lambda: 0.5) 
	#random.shuffle(labelfold, lambda: 0.5) #using the same shuffle 
	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices

"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set and call a function to get mean and variance of the train set.
It uses these to transform all datapoints of the test set. 
It returns the transformed test set. 

"""
def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	transformed = (testset - meantrain) / np.sqrt(variancetrain)
	return transformed


##############################################################################
#
#                      Cross validation SVM-C
#
##############################################################################

"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has dicts of all C's and gammas. For each combination it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (exept if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The lowest test average and the combination that produced it is returned with the train error rate.   
"""
def crossval(X_train, y_train, folds):
	# Set the parameters by cross-validation
	tuned_parameters = [{'gamma': [0.00000000001, 0.0000000001,0.000000001, 0.00000001, 0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}]
	
	labels_slices, features_slices = sfold(X_train, y_train, folds)
	accuracy = []

	#gridsearch
	for g in tuned_parameters[0]['gamma']:
		for c in tuned_parameters[0]['C']:
			temp = []
			tr_temp = []
			#crossvalidation
			for f in xrange(folds):
				crossvaltrain = []
				crossvaltrain_labels = []

				#define test-set for this run
				crossvaltest = np.array(features_slices[f])
				crossvaltest_labels = np.array(labels_slices[f])
				
				#define train set for this run
				for i in xrange(folds): #putting content of remaining slices in the train set 
					if i != f: # - if it is not the test slice: 
						for elem in features_slices[i]:
							crossvaltrain.append(elem) #making a list of trainset for this run
							
						for lab in labels_slices[i]:
							crossvaltrain_labels.append(lab) #...and a list of adjacent labels
				
				crossvaltrain = np.array(crossvaltrain)
				crossvaltrain_labels = np.array(crossvaltrain_labels)

				#Classifying using libsvm
				out = libsvm.fit(crossvaltrain, crossvaltrain_labels, svm_type=0, C=c, gamma=g)
				train_y_pred = libsvm.predict(crossvaltrain, *out)
				y_pred = libsvm.predict(crossvaltest, *out)

				#getting the train error count
				tr_count = 0
				for l in xrange(len(crossvaltrain_labels)):
					if train_y_pred[l] != crossvaltrain_labels[l]:
						tr_count +=1
				tr_temp.append(tr_count / len(crossvaltrain))

				#getting the test error count
				counter = 0
				for y in xrange(len(y_pred)):
					if y_pred[y] != crossvaltest_labels[y]:
						counter +=1
				#and storing the error result. 
				temp.append(counter / len(crossvaltest))

			#for every setting, get the average performance of the 5 runs:
			trainmean = np.array(np.mean(tr_temp))
			testmean = np.array(np.mean(temp))
			print "Average test error of %s: %.6f" %((c,g), testmean)
			accuracy.append([c,g,testmean, trainmean])

	#After all C's and gammas have been tried: get the best performance and the hyperparam pairs for that:
	accuracy.sort(key=itemgetter(2)) #sort by error - lowest first
	bestperf = accuracy[0][-2]
	besttrain = accuracy[0][-1]
	bestpair = tuple(accuracy[0][:2])
	print "\nBest hyperparameter (C, gamma)", bestpair
	return bestpair

"""
This function runs the SVM on the entire data set (training on the train set and evaluating on the testset)
using the best hyperparameter found from cross validation
It returns the train and test error
"""
def error_svc(X_train, y_train, X_test, y_test):
	out = libsvm.fit(X_train, y_train, svm_type=0, C=best_hyperparam_norm[0], gamma=best_hyperparam_norm[1])
	train_y_pred = libsvm.predict(X_train, *out)
	y_pred = libsvm.predict(X_test, *out)

	#train error
	c = 0
	for v in xrange(len(train_y_pred)):
		if y_train[v] != train_y_pred[v]:
			c +=1
	train_error = c / len(train_y_pred)

	#test error
	counter = 0
	for y in xrange(len(y_pred)):
		if y_pred[y] != y_test[y]:
			counter +=1
	test_error = counter / len(X_test)
	return train_error, test_error

# Getting train and test set

X_train, y_train = read_data(trainfile)
X_test, y_test = read_data(testfile)


##############################################################################
#
#                      PCA
#
##############################################################################


Galaxies = np.array([])
for d in X_train:
	if d[-1] == 0:
		np.append(Galaxies, d)

print Galaxies[:3]



##############################################################################
#
#                      Calling
#
##############################################################################




X_norm = meanfree(X_train)
X_trans = transformtest(X_train, X_test)

sigma = Jaakkola(X_train, y_train)
gamma = fromsigmatogamma(sigma)
print "Jaakkola sigma = ", sigma
print "Jaakkola gamma = ", gamma

norm_sigma = Jaakkola(X_norm, y_train)
norm_gamma = fromsigmatogamma(norm_sigma)
print "Norm sigma", norm_sigma
print "Norm gamma", norm_gamma


#best_hyperparam_raw = crossval(X_train, y_train, 5)
#best_hyperparam_norm = crossval(X_norm, y_train, 5)





