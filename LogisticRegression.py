import numpy as np
# make sure this class id compatable with sklearn's LogisticRegression
class LogisticRegression(object):

	def __init__(self, penalty='l2' , C=1.0 , max_iter=100 , verbose=0):
		# define all the model weights and state here
		pass

	def log_likelihood(features,target,weights): #function to maximise
		z=np.dot(features,weights)
		val=np.sum(target*scores - np.log(1+np.exp(scores)))
		return val

	def fit(self,X , Y):
		weights=np.zeros(X.shape[1])
		learning_rate=1e-5
		num_steps=2000
		for step in range(num_steps):
			z=np.dot(X,weights)
			predictions=1/(1+np.exp(-1*z))

			# gradient update in weights
			error=Y-predictions
			gradient=np.dot(X.T,error)
			weights+=learning_rate*gradient
			print 'im running',step
			#print log_likelihood(features,target,weights)
		self.model=weights
		print weights.shape
		print self.model.shape
		print self.model
		return self

	def predict(self,X ):
		val=np.dot(X,self.model)
		return np.round(1/(1+np.exp(-1*val)))
		# return a numpy array of predictions

	def score(self,X,y):
		return sum(self.predict(X)==y)/(len(y)*1.0)
