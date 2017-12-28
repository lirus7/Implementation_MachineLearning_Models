import numpy as np
# make sure this class id compatable with sklearn's GaussianNB
class GaussianNB(object):
    def __init__(self):
        pass

    def fit(self, X, y):
		#creating an array such that each class label with the datums belonging to it with their values
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
		#basicall getting the mean and stddev of each feature vecotor model[label]=[mean,stddev]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in separated])
        for i in range(len(self.model)):
			for j in range(len(self.model[i])):
				print self.model[i][j],i,j
        #print "finished"
        return self

    def prob(self, x, mean, std):
		#applying the gaussian formula
        print "im in prob"
        if(2 * std**2<0.0001):
            return 1
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        if(exponent<0.001):
            return 1
        #print exponent
		#returning the final value
        #print "im leaving prob"
        #print np.log(exponent / (np.sqrt(2 * np.pi) * std))
        #print exponent,std
        print exponent ,(np.sqrt(2 * np.pi) * std)
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
		#returning log probability of each class
        #print "im in log_prob"
        #print [[sum(self._prob(i, *s) for s, i in zip(summaries, x))for summaries in self.model] for x in X]
        return [[sum(self.prob(i, *s) for s, i in zip(summaries, x))for summaries in self.model] for x in X]

    def predict(self, X):
		#returns the index with with maximum log probability
        #print "im in pred"
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
		#simple function used to calculate accuracy
        return sum(self.predict(X) == y) / (len(y)*1.0)
