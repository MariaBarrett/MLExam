from __future__ import division
import numpy as np
import pylab as plt
from operator import itemgetter
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA

trainfile = open("VSTrain2014.dt", "r")
testfile = open("VSTest2014.dt", "r")

np.random.seed(99)
"""
This function reads in the files, strips by newline and splits by comma.
It shuffles the order of the datapoints before returning the dataset with the class label as the last value of each datapoint
"""
def read_data(filename):
	features = np.array([])
	labels = np.array([])
	entire = []
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(','),dtype='float')
		entire.append(l)
	np.random.shuffle(entire)
	for datapoint in entire:
		features = np.append(features, datapoint[:-1])
		labels = np.append(labels, datapoint[-1])
	features = np.reshape(features, (-1, len(datapoint)-1))
	return features, labels


##############################################################################
#
#                      KNN
#
##############################################################################

"""
This function gets the prior probability of each class
"""
def prior(y_train):
	sort = []
	count = Counter(y_train)
	c = 0.0
	for i in xrange(len(count)):
		sort.append(count[c] / len(y_train))
		c += 1.0
	sort = np.array(sort)
	return sort
"""
This function multiplies the prior probability with the predicted probability. 

def probawithprior(pred, proba):
	for i in xrange(len(proba)):
		pred[i] = pred[i] * proba[i]
	return pred

def one_vs_all_prep(y):
	newlabels = []
	for i in set(y):
		inner = []
		for label in y:
			if label == i:
				label = 1
			else:
				label = 0
			inner.append(label)
		newlabels.append(inner)
	return np.reshape(np.array(newlabels),(-1,(len(y))))

def one_vs_all_(X_train, y_train, X_test, y_test, k):
	newlabels = one_vs_all_prep(y_train)
	neigh = KNeighborsClassifier(n_neighbors=k)
	for i in xrange(len(newlabels):
		neigh.fit(X_train, newlabels[i])
		proba = neigh.proba(X_test, )  

This function calls KNN functions. I gets array of KNN from NearestNeighbor-function. 
Most frequent class is counted. 
1-0 loss and accuracy is calculated for train and test using counters. 
For the train accuracy I train on train and use datapoints from the same set.
For the test acc I train on train and use datapoints from test. 
"""	
def eval_knn(X_train, y_train, X_test, y_test, k):
	wrongtrain=0
	wrongtest=0
	#train set
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(X_train, y_train)
	y_pred_train = neigh.predict(X_train)
	y_pred = neigh.predict(X_test)
	#y_pred_train = probawithprior(neigh.predict_proba(X_train), prior)
	#y_pred = probawithprior(neigh.predict_proba(X_test), prior)
	
	for y in xrange(len(y_pred_train)):	
		if y_pred_train[y] != y_train[y]:
		#max_index = np.argmax(y_pred_train[y])
		#if float(max_index) != y_train[y]:
			wrongtrain +=1

	for y in xrange(len(y_pred)):	
		#max_index_ = np.argmax(y_pred[y])
		#if float(max_index_) != y_test[y]:
		if y_pred[y] != y_test[y]:
			wrongtest +=1
	return wrongtrain/len(y_train), wrongtest/len(y_test)

"""
This function calls LDA functions. 
1-0 loss and accuracy is calculated for train and test using counters. 
For the train accuracy I train on train and use datapoints from the same set.
For the test acc I train on train and use datapoints from test. 
"""	
def eval_lda(X_train, y_train, X_test, y_test):
	wrongtrain=0
	wrongtest=0
	#train set
	pri = prior(y_train)
	#clf = LDA(priors=pri)
	clf = LDA()
	clf.fit(X_train, y_train)
	y_pred_train = clf.predict(X_train)
	y_pred = clf.predict(X_test)
	
	for y in xrange(len(y_pred_train)):	
		if y_pred_train[y] != y_train[y]:
			wrongtrain +=1

	for y in xrange(len(y_pred)):	
		if y_pred[y] != y_test[y]:
			wrongtest +=1
	return wrongtrain/len(y_train), wrongtest/len(y_test)


"""
This function splits the shuffled train set in s equal sized splits. 
It expects the features, the labels and number of slices. 
It starts by making a copy of the labels and features and shuffles them. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints belonging to s.
"""
def sfold(features, labels, s):
	featurefold = np.copy(features)
	labelfold = np.copy(labels)

	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices


"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has dicts of all C's and gammas. For each combination it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (exept if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The lowest test average and the combination that produced it is returned with the train error rate.   
"""
def crossval(X_train, y_train, folds, function="KNN"):

	# Hyperparams: K
	Kcrossval = [1,3,5,7,9,11,13,15,17,21,25]
	print function
	labels_slices, features_slices = sfold(X_train, y_train, folds)
	cross_result = []
	for k in Kcrossval:	
				
		test_error = 0
		train_error = 0		
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
			if function == "KNN":
				acctrain, acctest = eval_knn(crossvaltrain, crossvaltrain_labels, crossvaltest, crossvaltest_labels, k)
			elif function == "LDA":
				pass
			train_error += acctrain
			test_error += acctest
		av_tr_error = train_error / folds
		av_te_error = test_error / folds
		print "Average 0-1 loss for k = %s: %.6f"%(k, av_te_error)

		cross_result.append([k, av_tr_error, av_te_error])

	cross_result.sort(key=itemgetter(-1)) #sort by error - lowest first
	bestperf = cross_result[0][-1]
	besttrain = cross_result[0][-2]
	bestk = cross_result[0][0]
	print "\nBest k %s, train error = %.6f, test error = %.6f" %(bestk, besttrain, bestperf)
	return bestk


########################################################################################
#
#								Normalizing
#
########################################################################################

#Computing mean and variance
"""
This function takes a dataset and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	mean = sum(data) / len(data)
	variance = sum((data - mean)**2) / len(data)
	return mean, variance


"""
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set is returned
"""
def meanfree(data):	
	mean, variance = mean_variance(data)
	meanfree = (data - mean) / np.sqrt(variance)
	return meanfree

def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	transformed = (testset - meantrain) / np.sqrt(variancetrain)
	return transformed



X_train, y_train = read_data(trainfile)
X_test, y_test = read_data(testfile)

X_norm = meanfree(X_train)
X_trans = transformtest(X_train, X_test)

train_mean, train_var = mean_variance(X_train)
print "Mean of raw trainset", train_mean
print "Variance of raw trainset", train_var

trans_mean, trans_var = mean_variance(X_trans)
print "Mean for transformed testset", trans_mean
print "Variance of transformed testset", trans_var

print '*'*45
print 'Raw 5-fold cross validation'
print '*'*45
bestk = crossval(X_train, y_train, 5)

print '*'*45
print 'Normalized 5-fold cross validation'
print '*'*45
bestk_norm = crossval(X_norm, y_train,5)

trainerr, testerr = eval_knn(X_norm, y_train, X_trans, y_test, bestk_norm)
print "Trained on training set and evaluated on test set with best-k: %s. Train error = %.4f, test error = %.4f" %(bestk_norm, trainerr, testerr)

print '*'*45
print 'Raw LDA'
print '*'*45
trainerr, testerr = eval_lda(X_train, y_train, X_test, y_test,)
print "Trained on train set and evaluated on test set. Train error = %.4f, test error = %.4f" %(trainerr, testerr)

