from __future__ import division
import numpy as np
import pylab as plt
from operator import itemgetter
from collections import Counter

trainfile = open("VSTrain2014.dt", "r")
testfile = open("VSTest2014.dt", "r")

k = 1
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
	features = np.reshape(entire, (-1, len(entire[0])))
	return entire


##############################################################################
#
#                      KNN
#
##############################################################################

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

def NearestNeighbor(tr,ex0):
	"""
  	This function expects a dataset and a datapoint. 
  	It calls the euclidean and stores the distances to the solo datapoint of all datapoints in the dataset. These are stores in a list of lists. 
  	This lists are sorted according to distances and K-nearest datapoints are returned 
	"""
	distances = []

	#distances.append(ex0)
	for ex in tr:
		curr_dist = euclidean(ex,ex0) 
		distances.append([curr_dist,ex])

	distances.sort(key=itemgetter(0))
	KNN = distances[:k] #taking only the k-best matches
	return KNN


"""
This function calls KNN functions. I gets array of KNN from NearestNeighbor-function. 
Most frequent class is counted. 
1-0 loss and accuracy is calculated for train and test using counters. 
For the train accuracy I train on train and use datapoints from the same set.
For the test acc I train on train and use datapoints from test. 
"""	
def eval(train, test):
	wrongtrain=0
	wrongtest=0
	#train set
	for ex in train:
		ex_prime=NearestNeighbor(train,ex)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn) #result is a list of the k-nearest classes
		bestresult = result.most_common(1) #majority vote
		if bestresult[0][0] != ex[-1]:
			wrongtrain +=1

	#test set		
	for ex in test:
		ex_prime=NearestNeighbor(train,ex)
		knn =[]
		for elem in ex_prime:
			knn.append(elem[-1][-1]) #that's the class
			result = Counter(knn)
		result = result.most_common(1)
		if result[0][0] != ex[-1]:
			wrongtest +=1
	return wrongtrain/len(train), wrongtest/len(test)

"""
This function splits the shuffled train set in s equal sized splits. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints.
"""
def sfold(data,s):
	slices = [data[i::s] for i in xrange(s)]
	return slices


"""
After having decorated, this function gets a slice for testing and uses the rest for training.
First we choose test-set - that's easy.
Then for every test-set for as many folds as there are: use the remaining as train sets exept if it's the test set. 
Then we sum up the result for every run and average over them and print the result.  
"""
def crossval(trainset, folds):
	print '*'*45
	print '%d-fold cross validation' %folds
	print '*'*45
	cross_result = []
	slices = sfold(trainset,folds)
	Kcrossval = [1,3,5,7,9,11,13,15,17,21,25]

	for k in Kcrossval:
		print "Number of neighbors \t%d" %k
		temp = []
		temp = 0
		temptrain = 0
		for f in xrange(folds):
			crossvaltest = slices[f]
			crossvaltrain =[]
			
			for i in xrange(folds):
				if i != f: 
					for elem in slices[i]:
						crossvaltrain.append(elem) #making a new list of crossvaltrains
			acctrain, acctest = eval(crossvaltrain,crossvaltest)
			temp += acctest	
			temptrain += acctrain
		av_tr_result = temptrain / folds
		av_result = temp/folds
		print "Averaged 0-1 loss \t%1.4f" %av_result
		print "-"*45
		temp.append(k, av_tr_result, av_result)
		cross_result.append(temp)
	cross_result.sort(key=itemgetter(-1))
	bestk = cross_result[0][0]
	bestacc = cross_result[0][-1]
	bestrain = cross_result[0][1]
	print "Best k = %s. Test error = %.4f. Train error = %.4f" %(bestk, bestacc, besttrain)
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
	Mean = []
	Variance = []
	number_of_features = len(data[0]) - 1 #Leaving out the class
	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s +=elem[i]
		mean = s / len(data)
		Mean.append(mean)
		
		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(variance)
	return Mean, Variance


"""
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set is returned
"""
def meanfree(data):
	number_of_features = len(data[0]) - 1 #Leaving out the class

	mean, variance = mean_variance(data)
	print "Mean", mean
	print "Variance", variance

	new = np.copy(data)

	for num in xrange(number_of_features):
		for i in xrange(len(data)):
			#replacing at correct index in the copy
			new[i][num] = (data[i][num] - mean[num]) / np.sqrt(variance[num])
	return new

def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)

	number_of_features = len(trainset[0]) - 1 #Leaving out the class

	newtest = np.copy(testset)

	for num in xrange(number_of_features):
		for i in xrange(len(testset)):
			#replacing at correct index in the copy
			newtest[i][num] = (testset[i][num] - meantrain[num]) / np.sqrt(variancetrain[num])
	return newtest

trainset = read_data(trainfile)
testset = read_data(testfile)

trainset = trainset[::10]

train_norm = meanfree(trainset)
test_trans = transformtest(testset)

bestk = crossval(trainset)
bestk_norm = crossval(train_norm)

