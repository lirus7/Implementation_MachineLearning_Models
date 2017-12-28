import numpy as np
# make sure this class id compatable with sklearn's DecisionTreeClassifier
class DecisionTreeClassifier(object):

	def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
		# define all the model weights and state here
		pass

	def partition(self,a):
	    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

	def entropy(self,attribute):
		res=0
		val,freq=np.unique(attribute,return_counts=True)
		freq=freq.astype('float')/len(attribute)
		for i in freq:
			res-=i*np.log2(i)
		return res

	#y is the original set and x is the attribute we find the confitional entropy

	def information_gain(self,y,x):
		res=self.entropy(y)#particular attribute  to pass
		val,freq=np.unique(x,return_counts=True)
		freq=freq.astype('float')/len(x)
		#subtracting the complete residue
		for p,v in zip(freq,val):
			res-=p*self.entropy(y[x==v])

		return res

	def split(self,x,y):
		if len(set(y))==1 or len(y)==0: #pure thingy
			return y
		#finding the best attribute for gain
		gain=np.array([self.information_gain(y,x_attr)for x_attr in x.T])
		selected_attr=np.argmax(gain)

		if(np.all(gain<0.00001)):
			return y
		res={}
		sets=self.partition(x[:,selected_attr])
		#print "im items",sets.items()
		#print "im set",sets
		for k,v in sets.items():
			x_subset=x.take(v,axis=0)
			y_subset=y.take(v,axis=0)
			if(selected_attr not in self.unique):
				self.unique.append(selected_attr)
			res[(selected_attr,k)]=self.split(x_subset,y_subset)
		return res

	def fit(self,X , Y):
		self.unique=[]
		self.model=self.split(X,Y)
		return self
		# return a numpy array of predictions

	def predict(self,X,y):
		ans=[]
		for i in X:
			q=0
			temp=self.model[(self.unique[0],i[self.unique[0]])]
			while 'dict' in str(type(temp)):
				for j in temp.keys():
					val=j[0]
					break
				if (val,i[val]) in temp.keys():
					temp=temp[(val,i[val])]
					continue
				else:
					q=1
					break
			if(q==1):
				ans.append(-1)
			else:
				ans.append(temp)
		counter=0
		for i in range(len(y)):
			if(y[i]==ans[i][0]):
				counter=counter+1
		listx=[]
		for i in range(len(ans)):
			listx.append(ans[i][0])
		return counter,listx

	def score(self,X,y):
		return self.predict(X,y)[0]/(len(y)*1.0)
