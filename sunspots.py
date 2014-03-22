from __future__ import division
import numpy as np
import pylab as plt

""" REGRESSION: SUNSPOT PREDICTION """

"""II.2.1 MAXIMUM LIKELIHOOD SOLUTION """

""" Load the data from the train and test files """
def load_files(filename):
	dataset = np.array([])
	f = open(filename)

	for l in f.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		dataset = np.append(dataset, l)	

	dataset = np.reshape(dataset, (-1,len(l))) #reshape the np array to correct shape

	return dataset

""" 
Get the subset of observation variables
ds = dataset
idx = the array indexes in ds to get as subset 
"""
def getSubset(ds, idx):
	return ds[:,idx]

""" 
Create the design matrix from vector X - either for Linear or quadratic basis functions
- Linear design matrices would look like this: [1, x1, ..., xn]
- Quadratic design matrices would look like this: (e.g. with 3 variables)
[1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3] for each row 
"""
def createDesignMatrix(X, type="linear"):
	#if type=="linear, create an array of size 1+n, 
	#else if type=="quadratic", create an array of 1 + n + n + (summation of X0..Xn-1)
	l = len(X[0])
	size = 1 + l

	if (type=="quadratic"):
		size += l + (((l-1)*l) / 2)

	phi = np.zeros((len(X), size))

	#set first column values to 1
	phi[:,0] = 1 

	for c in range(l):
		phi[:,c+1] = X[:,c] # phi(x) = x

		if (type=="quadratic"):
			phi[:,c+1+l] = X[:,c] ** 2 # phi(x) = x**2

	if (type=="quadratic"):
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
def predict(w, X_test):
	phi = createDesignMatrix(X_test) #create design matrix from the test variables
	y = np.zeros(len(phi)) # predicted classes

	#summate over all rows in phi and save class vector in y
	for i in range(len(phi)):
		sum = 0
		for j in range(len(phi[i])):
			sum += w[j] * phi[i][j]
		y[i] = sum

	return y

""" 
This function calculates the Root Mean Square
sqrt( 1/N * sum(tn-yn))
t = actual value from the dataset
y = predicted value 
"""
def calculateRMS(t, y):
	N = len(t)
	sum = 0
	
	for n in range(N):
		sum += (t[n] - y[n])**2

	RMS = np.sqrt(sum/N)
	return RMS


""" II.2.2 MAXIMUM A POSTERIORI SOLUTION """
def computeBayesianMeanAndCovariance(X, t, alpha):
	beta = 1
	#get design matrix
	phi = createDesignMatrix(X)

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


def run():
	""" MAIN """
	train = load_files("sunspotsTrainStatML.dt")
	test = load_files("sunspotsTestStatML.dt")

	subsets = [[2,3],[4],[0,1,2,3,4]]

	#get subset of observation variables
	x1 = getSubset(train, subsets[0])
	x2 = getSubset(train, subsets[1])
	x3 = getSubset(train, subsets[2])

	test_x1 = getSubset(test, subsets[0])
	test_x2 = getSubset(test, subsets[1])
	test_x3 = getSubset(test, subsets[2])

	#get vector of all target variables
	t = getSubset(train, [5])
	test_t = getSubset(test, [5])

	#build design matrix
	phi1 = createDesignMatrix(x1)
	phi2 = createDesignMatrix(x2)
	phi3 = createDesignMatrix(x3)

	#get weight vectors
	w1 = findML(phi1, t)
	w2 = findML(phi2, t)
	w3 = findML(phi3, t)

	#get predicted target classes on test
	y1 = predict(w1, test_x1)
	y2 = predict(w2, test_x2)
	y3 = predict(w3, test_x3)

	#plot x & t for variable selection 2
	plt.plot(t, x2, "ro", label="x vs training label")
	plt.plot(test_t, test_x2, "bo", label="x vs actual test label")
	plt.plot(y2[0], test_x2[0], y2[:-1], test_x2[:-1], "k-")
	plt.plot(y2, test_x2, "go", label="x vs predicted test label")
	plt.ylabel("x = sunspots in year s-16")
	plt.xlabel("t = sunspots in year s")
	plt.legend(loc="best")
	plt.show()

	#calculate Root Mean Square (RMS) for each variable selection
	RMS1 = calculateRMS(test_t, y1)
	RMS2 = calculateRMS(test_t, y2)
	RMS3 = calculateRMS(test_t, y3)
	#Print out the results
	print "RMS with D=%d = %f"%(len(subsets[0]), RMS1)
	print "RMS with D=%d = %f"%(len(subsets[1]), RMS2)
	print "RMS with D=%d = %f"%(len(subsets[2]), RMS3)

	years = np.arange(96) + 1916 #all the years from 1916-2011 are in the test dataset
	plt.plot(years, test_t, "xg-", label="Actual")
	plt.plot(years, y1, "xr-", label="D=2")
	plt.plot(years, y2, "xb-", label="D=1")
	plt.plot(years, y3, "xy-", label="D=5")
	plt.xlabel("years")
	plt.ylabel("sunspots")
	plt.legend(loc="best")
	plt.show()

	""" Bayesian LR - MAIN """
	alphas = np.arange(0, 160, 5)
	bys_RMS1, bys_RMS2, bys_RMS3 = np.array([]), np.array([]), np.array([])

	for alpha in alphas:
		bysMean, bysCovariance = computeBayesianMeanAndCovariance(x1, t, alpha)
		bys_y1 = predict(bysMean, test_x1)

		bysMean, bysCovariance = computeBayesianMeanAndCovariance(x2, t, alpha)
		bys_y2 = predict(bysMean, test_x2)

		bysMean, bysCovariance = computeBayesianMeanAndCovariance(x3, t, alpha)
		bys_y3 = predict(bysMean, test_x3)

		#calculate Root Mean Square (RMS) for each variable selection
		bys_RMS1 = np.append(bys_RMS1, calculateRMS(test_t, bys_y1))
		bys_RMS2 = np.append(bys_RMS2, calculateRMS(test_t, bys_y2))
		bys_RMS3 = np.append(bys_RMS3, calculateRMS(test_t, bys_y3))

	plt.plot(alphas, bys_RMS1, ".r-", label="Bayes D=2")
	plt.plot(alphas, bys_RMS2, ".b-", label="Bayes D=1")
	plt.plot(alphas, bys_RMS3, ".g-", label="Bayes D=5")
	plt.plot(alphas, [RMS3] * len(bys_RMS3), ".y-", label="ML D=5")
	plt.xlabel("alphas")
	plt.ylabel("Bayesian Root Mean Square")
	plt.legend(loc="best")

	plt.show()

run()