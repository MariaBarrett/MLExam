from __future__ import division
import numpy as np
import random
from sklearn.svm import SVC
import operator
from collections import Counter
import pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(99)


trainfile = open("SGTrain2014.dt", "r")
testfile = open("SGTest2014.dt", "r")

"""
This function reads in the files, strips by newline and splits by comma.
It shuffles the order of the datapoints before separating the labels (last value) from the features (the rest).  
It returns the labels as a 1D list and the features as one numpy array per row.
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
				temp.append(eu) #temp is a list of eucidean distances between other-class datapoints
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

	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices



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
	tuned_parameters = {'gamma': [0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}
	
	labels_slices, features_slices = sfold(X_train, y_train, folds)
	accuracy = []

	#gridsearch
	for g in tuned_parameters['gamma']:
		for c in tuned_parameters['C']:
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

				#Classifying using library function
				clf = SVC(C=c, gamma=g)
				clf.fit(crossvaltrain, crossvaltrain_labels)
				train_y_pred = clf.predict(crossvaltrain)
				y_pred = clf.predict(crossvaltest)
				
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
	accuracy.sort(key=operator.itemgetter(2)) #sort by error - lowest first
	bestperf = accuracy[0][-2]
	besttrain = accuracy[0][-1]
	bestpair = tuple(accuracy[0][:2])
	print "\nBest hyperparameter (C, gamma) %s, train error = %.6f, test error = %.6f" %(bestpair, besttrain, bestperf)
	return bestpair


def error(y_pred, y):
	c = 0
	for i in xrange(len(y)):
		if y_pred[i] != y[i]:
			c += 1
	return c / len(y)  

"""
This function runs the SVM on the entire data set (training on the train set and evaluating on the testset)
using the best hyperparameter found from cross validation. It expects the hyperparamester as a tuple (C, gamma)
It returns the train and test error
"""
def error_svc(X_tr, y_tr, X_te, y_te, best_hyper):
	clf = SVC(C=best_hyper[0], gamma=best_hyper[1])
	clf.fit(X_tr, y_tr)
	#model = libsvm.fit(X_tr, y_tr, C=best_hyper[0], gamma = best_hyper[1])
	train_y_pred = clf.predict(X_tr)
	y_pred = clf.predict(X_te)
	#train_y_pred = libsvm.predict(X_tr, *model)
	#y_pred = libsvm.predict(X_te, *model)

	#train error
	train_error = error(train_y_pred, y_tr)
	#test error
	test_error = error(y_pred, y_te)
	return train_error, test_error

# Getting train and test set

X_train, y_train = read_data(trainfile)
X_test, y_test = read_data(testfile)

#X_train = X_train[::10]
#y_train = y_train[::10]
##############################################################################
#
#                      PCA
#
##############################################################################

#Making an array of all training datapoints with label 0
def getGalaxies(X,y):	
	Galax = []
	for i,j in enumerate(y):
		if j == 0.0:
			Galax.append(X[i])	
	return np.array(Galax)


def princomp(galax):
	pca = PCA(copy=True)
	transformed = pca.fit_transform(galax)
	components = pca.components_

	M = (galax-np.mean(galax.T,axis=1)).T # subtract the mean (along columns)
 	eigv, eigw = np.linalg.eig(np.cov(M)) 

 	print np.cov(M)
 	x = [1,2,3,4,5,6,7,8,9,10]

	plt.plot(x, eigv)
	plt.title("Eigenspectrum")
	plt.ylabel("Eigenvalue")
	plt.xlabel("Components")
	plt.show()

def princomp2(galax):
	clustermeans = mykmeans(galax) #getting 10D clustermeans from the normalized dataset
	print "*" * 45
	print "K-means clustering"
	print "*" * 45
	print "k = 2"
	print "Mean of clusters:", clustermeans

	pca2 = PCA(n_components=2)
	pca2.fit(galax)
	transformed = pca2.transform(galax)

	plt.title("Data projected on the first two principal components")
	plt.xlabel("first principal component")
	plt.ylabel("second principal component")
	plt.plot([x[0] for x in transformed], [x[1] for x in transformed], 'bx', label = "Galaxies")
	plt.legend(loc='upper right')
	plt.show()	

	transformed_mean1 = pca2.transform(clustermeans[0])
	transformed_mean2 = pca2.transform(clustermeans[1])	

	meanx = [transformed_mean1[0][0], transformed_mean2[0][0]]
	meany = [transformed_mean1[0][1], transformed_mean2[0][1]]

	plt.title("Data projected on the first two principal components with transformed clustermeans")
	plt.xlabel("first principal component")
	plt.ylabel("second principal component")
	plt.plot([x[0] for x in transformed], [x[1] for x in transformed], 'bx', label = "Galaxies")
	plt.plot(meanx, meany, 'ro', label = "Cluster means")
	plt.legend(loc='upper right')
	plt.show()
##############################################################################
#
#                     K-means
#
##############################################################################
k = 2
"""
This function expects a dataset and a list of centers. It calls the euclidean function and
stores the distance to each center in a list. From that list it gets the index of the minimum value
and stores the datapoint in a new list of lists at that index.
It returns the list of lists of the reordered datapoints.
Then itcalculates the mean for the k lists in the main list.
It returns a list of lists with the calculated means. 
"""

def assigning_to_center(dataset, initial):
	sorted_according_to_center = [[],[]] 
	assert len(initial) == k
	
	for datapoint in dataset:
		distance = []
		for i in xrange(len(initial)):
			d = euclidean(datapoint, initial[i]) 
			distance.append(d)
		min_index, min_value = min(enumerate(distance),key=operator.itemgetter(1)) #finding index of min value
		sorted_according_to_center[min_index].append(datapoint) #...and append in new list at that index

	listofmeans = []
	for sublist in sorted_according_to_center:
		submean = sum(sublist) / len(sublist)
		listofmeans.append(submean)
	return listofmeans


"""
Here is where everything is put together:
I have assigned k fixed datapoints to be my initial centers.
I assign datapoints to them and calculate mean1.
I assign to this mean and calculates mean1.

Until mean1 and mean2 converge, I continue assigning and calculating new means
"""
def mykmeans(standardized_data):
	initial = [standardized_data[98], standardized_data[99]]
	mean1 = assigning_to_center(standardized_data, initial)
	mean2 = assigning_to_center(standardized_data, mean1)
	
	while not np.array_equal(mean1, mean2):
		#while len(set(mean1[i]).intersection(mean2[i])) < len(mean1[i]): #as long as they are not identical
			mean1 = assigning_to_center(standardized_data, mean2)
			mean2 = assigning_to_center(standardized_data, mean1)
	return mean1

##############################################################################
#
#                      Calling SVM
#
##############################################################################

X_norm = meanfree(X_train)
X_trans = transformtest(X_train, X_test)


def run_svc():
	mean_train, var_train = mean_variance(X_train)
	print "mean of train set:", mean_train
	print "Variance train set:", var_train 

	mean_trans, var_trans = mean_variance(X_trans)
	print "mean of transformed test set:", mean_trans
	print "Variance of transformed test set:", var_trans

	sigma = Jaakkola(X_train, y_train)
	gamma = fromsigmatogamma(sigma)
	print "Jaakkola sigma = ", sigma
	print "Jaakkola gamma = ", gamma

	norm_sigma = Jaakkola(X_norm, y_train)
	norm_gamma = fromsigmatogamma(norm_sigma)
	print "Norm sigma", norm_sigma
	print "Norm gamma", norm_gamma

	print '*'*45
	print "Raw"
	print '*'*45
	print '-'*45
	print '5-fold cross validation'
	print '-'*45
	print '(C, gamma)'
	best_hyperparam_raw = crossval(X_train, y_train, 5)

	print '*'*45
	print "Normalized"
	print '*'*45
	print '-'*45
	print '5-fold cross validation'
	print '-'*45
	print 'C, gamma'
	best_hyperparam_norm = crossval(X_norm, y_train, 5)
	print best_hyperparam_norm

	tr_error, te_error = error_svc(X_train, y_train, X_test, y_test, best_hyperparam_norm)
	tr_error_norm, te_error_norm = error_svc(X_norm, y_train, X_trans, y_test, best_hyperparam_norm)

	print "-"*45
	print "Trained on train set and evaluated on test set"
	print "-"*45
	print "Best hyperparameter from normalized: (C, gamma)", best_hyperparam_norm
	print "Raw: train error= %.4f, test error = %.4f" %(tr_error, te_error)
	print "Normalized: train error= %.4f, test error = %.4f" %(tr_error_norm, te_error_norm)

#run_svc()


##############################################################################
#
#                      Calling PCA
#
##############################################################################

Galaxies = getGalaxies(X_train, y_train)
Galaxies_norm = meanfree(Galaxies) #normalizing Galaxies to 0 mean uni variance

princomp(Galaxies_norm)
princomp2(Galaxies_norm)


#Find a way to plot 10D on the 2D plot
"""
tuned_parameters = [{'gamma': [0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}]
	


clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
clf.fit(X_train, y_train)

print "Best parameters set found on development set:"
print clf.best_estimator_

print"Grid scores on development set:"
print ""
for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)
print ""


clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
clf.fit(X_norm, y_train)

print "Best parameters set found on development set:"
print clf.best_estimator_

#print"Grid scores on development set:"
print ""
#for params, mean_score, scores in clf.grid_scores_:
#    print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

best = SVC(C=100, gamma=0.1)
best.fit(X_train, y_train)
y_pred_train = best.predict(X_train)
y_pred = best.predict(X_test)

best.fit(X_norm, y_train)
y_pred_tr_norm = clf.predict(X_norm)
y_pred_trans = clf.predict(X_trans)

print "Raw"
print "Train", error(y_pred_train, y_train)
print "Test", error(y_pred, y_test)

print "Normalized"
print "Train", error(y_pred_tr_norm, y_train)
print "Test", error(y_pred_trans, y_test)

"""
