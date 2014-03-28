from __future__ import division
import numpy as np
import random
import pylab as plt
import scipy.optimize as optimize
from sklearn.linear_model import LinearRegression
import operator
from multipolyfit import multipolyfit as mpf


trainfile = open("SSFRTrain2014.dt", "r")
testfile = open("SSFRTest2014.dt", "r")
np.random.seed(1)
"""
This function reads in the files, strips by newline and splits by space char. 
It returns the labels as a 1D list and the features as one numpy array per row.
"""
def read_data(filename):
	features = np.array([])
	labels = np.array([])
	entire = []
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		entire.append(l)
	np.random.shuffle(entire)	
	for datapoint in entire:
		features = np.append(features, datapoint[:-1])
		labels = np.append(labels, datapoint[-1])
	features = np.reshape(features, (-1, len(datapoint)-1))
	return features, labels

X_train, y_train = read_data(trainfile)
X_test, y_test = read_data(testfile)
"""
clf = LinearRegression()
clf.fit(X_train, y_train)
y_pred_Sk = clf.predict(X_test)


Create the design matrix from vector X - either for Linear or quadratic basis functions
- Linear design matrices would look like this: [1, x1, ..., xn]
- Quadratic design matrices would look like this: (e.g. with 3 variables)
[1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3] for each row 
"""
def createDesignMatrix(X, deg, type="linear"):
	#if type=="linear, create an array of size 1+n, 
	#else if type=="poly", create an array of 1 + n + n + (summation of X0..Xn-1)
	l = len(X[0])
	size = 1 + l

	if (type=="poly"):
		size += l + (((l-1)*l) / 2)
	phi = np.zeros((int(len(X)), int(size)))

	#set first column values to 1
	phi[:,0] = 1 

	for c in range(l):
		phi[:,c+1] = X[:,c] # phi(x) = x

		if (type=="poly"):
			phi[:,c+1+l] = X[:,c] ** deg # phi(x) = x**2

	if (type=="poly"):
		# phi(x) = x1 * x2 ... for (n-1)!
		j = 0
		for c in xrange(0, l-1):
			for i in xrange(c+1, l):
				phi[:, j+1+l+l] = X[:, c] * X[:, i]
				j += 1

	return phi


""" 
Finding the Maximum Likelihood 
phi = design matrix PHI 
t = vector of corresponding target variables 
"""
def findML(phi, t):
	wML = np.dot(phi.T, phi)
	wML = np.linalg.pinv(wML)
	wML = np.dot(wML, phi.T)
	wML = np.dot(wML, t)
	return wML

""" 
In this function we will use the weight vectors from the training set and use them on the test set's variables
w = weight vectors from the train dataset 
X_test = the subset of X variables from the test set 
"""
def predict(w, X_test, deg, phitype):
	phi = createDesignMatrix(X_test, deg, type=phitype) #create design matrix from the test variables
	y = np.zeros(len(phi)) # predicted classes

	#summate over all rows in phi and save class vector in y
	for i in range(len(phi)):
		s = 0
		for j in range(len(phi[i])):
			s += w[j] * phi[i][j]
		y[i] = s
	return y

""" 
This function calculates the Mean Squared Error
t = actual value from the dataset
y = predicted value 
"""
def calculateMSE(t, y):
	N = len(t)
	sum = 0
	for n in range(N):
		sum += (t[n] - y[n])**2
	MSE = sum/N
	return MSE

""" This function computes the mean and covariance using maximum aposteriori. """
def computeBayesianMeanAndCovariance(X, t, alpha, deg):
	beta = 1
	#get design matrix
	phi = createDesignMatrix(X, deg, type="poly")

	#get second part of covariance matrix
	bpp = beta * np.dot(phi.T, phi)

	#get first part of covariance matrix
	aI = np.zeros(bpp.shape)
	np.fill_diagonal(aI, alpha) #alpha * I

	covariance = aI + bpp
 	
 	#get each part of the mean equation
 	bs = beta * np.linalg.pinv(covariance)
 	mean = np.dot(bs, np.dot(phi.T, t))
 	mean = mean.reshape(-1,len(mean))[0]

 	return mean, covariance  

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

print "*"*45
print "Linear regression"
print "*"*45

phi = createDesignMatrix(X_train, 1, type="linear")
weights = findML(phi, y_train)
print "Bias:", weights[0]
y_pred_train = predict(weights, X_train, 1, "Linear")
y_pred = predict(weights, X_test, 1, "Linear")
MSE_train = calculateMSE(y_train, y_pred_train)
MSE_test = calculateMSE(y_test, y_pred)
print "MSE Trains", MSE_train
print "MSE Test", MSE_test


##############################################################################
#
#                    	Polynomial regression
#
##############################################################################

print "*"*45
print "Polynomial regression"
print "*"*45

def alpha_deg_gridsearch(X_tr, y_tr, X_te, y_te):
	folds = 5
	alphas = np.arange(0, 25, 1)
	degrees = [2,3,4]
	results = []
	labels_slices, features_slices = sfold(X_train, y_train, folds)
	for deg in degrees: 
		for alpha in alphas:
			te_temp = 0
			tr_temp = 0
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

				Mean, Covariance = computeBayesianMeanAndCovariance(crossvaltrain, crossvaltrain_labels, alpha, deg)
				y_pred_tr = predict(Mean, crossvaltrain, deg, "poly")
				y_pred = predict(Mean, crossvaltest, deg, "poly")
				#calculate Root Mean Square (RMS) for each variable selection
				MSE_tr = calculateMSE(y_pred_tr, crossvaltrain_labels)
				MSE_te = calculateMSE(y_pred, crossvaltest_labels)
				
				tr_temp += MSE_tr
				te_temp += MSE_te
			tr_temp = tr_temp / folds
			te_temp = te_temp / folds
			print "Degree = %s, alpha = %s,  av. train MSE = %.6f, av. test MSE = %.6f" %(deg, alpha, tr_temp, te_temp)
			results.append([tr_temp, te_temp, (deg, alpha)])

	resultsort = results.sort(key=operator.itemgetter(1)) #sort by error - lowest first
	print resultsort
	bestdeg_alpha = results[0][-1]
	train_MSE = results[0][0]
	test_MSE = results[0][1]
	print "Best (degree, alpha): %s, train MSE = %.6f, test MSE = %.6f " %(bestdeg_alpha, train_MSE, test_MSE)
	return bestdeg_alpha

print "-"*45
print "Gridsearch to find best degree and alpha"
print "-"*45

deg_alpha = alpha_deg_gridsearch(X_train, y_train, X_test, y_test) 

Mean, Covariance = computeBayesianMeanAndCovariance(X_train, y_train, 4.5, deg_alpha[0])
y_pred_tr = predict(Mean, X_train, deg_alpha[0], "poly")
y_pred = predict(Mean, X_test, deg_alpha[0], "poly")
MSE_tr = calculateMSE(y_pred_tr, y_train)
MSE_te = calculateMSE(y_pred, y_test)
print "Trained on train set and evaluated on test set with best hyperparameter pair (degree, alpha), %s. Train MSE = %.6f. Test MSE = %.6f" %(deg_alpha, MSE_tr, MSE_te)
"""
#Regression
degrees = [1,2,3,4,5]
for deg in degrees: 
	model = mpf(X_train, y_train, deg=deg, model_out=True)
	beta, powers = mpf(X_train, y_train, deg=deg, powers_out=True)

	y_pred = []
	for d in X_test:
		y = model(d[0],d[1],d[2],d[3])
		y_pred.append(y)
	y_pred = np.array(y_pred)

	y_pred_train = []
	for d in X_train:
		y = model(d[0],d[1],d[2],d[3])
		y_pred_train.append(y)
	y_pred_train = np.array(y_pred_train)

	MSE_train = calculateMSE(y_pred_train, y_train)
	MSE_test = calculateMSE(y_pred, y_test)
	print "Degree: %s, Train MSE = %.6f, Test MSE = %.6f" %(deg, MSE_train, MSE_test)
"""

