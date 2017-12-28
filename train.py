import os
import os.path
import argparse
import h5py
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from Models.GaussianNB import GaussianNB
import Models.LogisticRegression
import Models.DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.externals import joblib


def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y

def calc_factors(number):
	factor_list=[]
	for i in range(2,number//2):
		if(number%i==0):
			factor_list.append(i)
	return factor_list

# Preprocess data and split it

def k_cross_validation(X,label_list,j,k):
	test_data=X[j:(j+1)*k]
	test_label=label_list[j:(j+1)*k]
	training_data=[]
	training_label=[]
	training_data.extend(X[0:j])
	training_data.extend(X[(j+1)*k:])
	training_label.extend(label_list[0:j])
	training_label.extend(label_list[(j+1)*k:])
	return test_data,test_label,training_data,training_label


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
training_label=[]
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()

# Load the test data
X,Y=load_h5py(args.train_data)
if(args.train_data=='Data/part_A_train.h5'):
	t=1;
elif(args.train_data=='Data/part_B_train.h5'):
	t=2;
else:
	t=3;

np.asarray(X)
np.asarray(Y)
label_list=[]

X_y,X_x=X.shape
Y_y,Y_x=Y.shape
# Generating the labels in a single list to plot

for i in range(X_y):
    for j in range(Y_x):
        if Y[i][j]==1:
            label_list.append(j)
            break
print label_list
print X.shape
print Y.shape


# Train the models

if args.model_name == 'GaussianNB':
	#No hyperparameters
	clf=GaussianNB()
	clf.fit(X, label_list)
	#Doing k cross validation
	listx=calc_factors(X_y)
	best=0
	itera=0
	factorval=0
	count=0
	for i in range(len(listx)):
		k=listx[len(listx)-1-i]
		factor=X_y//k
		for j in range(factor-1):
			test_data,test_label,training_data,training_label=k_cross_validation(X,label_list,j,k)
#			print len(test_data),len(test_label),len(training_data),len(training_label)
			clf=GaussianNB()
			clf.fit(training_data,training_label)
			score=clf.score(test_data,test_label)
			print listx[i],j,score
			if score>best:
				best=score
				itera=j
				best_model=clf
				factorval=listx[i]
		count=count+1
		if(count>10):
			break;
	best_k_val=factorval
	print best,itera,factorval
	joblib.dump(best_model, 'Weights/Gaussian{}.pkl'.format(t))

elif args.model_name == 'LogisticRegression':
	best_regularization_strength=0
	best_c_value=0
	c=[0.1,0.2,0.3,0.4,0.5,1]
	reg_strength=[i for i in range(1,11)]
	accr_val=[]
	best_score=0
	for i in range(len(c)):
		for j in range(1,11):
			logreg=LogisticRegression(C=c[i],max_iter=j)#C is supposed to be used but the value seems to be same
			logreg.fit(X, label_list)
			grid_score=logreg.score(X,label_list)
			if(i==0):
				accr_val.append(grid_score)
			print i,j
			if(grid_score>best_score):
				best_regularization_strength=j
				best_c_value=i
				best_score=grid_score
				best_model=logreg
	print "The best score obtained for changing hyperparameter ",best_regularization_strength, best_c_value,best_score
	plt.scatter(reg_strength,accr_val)
	plt.savefig('Plots/result_logistic{}'.format(t))
	joblib.dump(best_model, 'Weights/Logistic{}.pkl'.format(t))
	plt.show()
	listx=calc_factors(X_y)
	best=0
	itera=0
	factorval=0
	for i in range(len(listx)):
		k=listx[i]
		factor=X_y//k
		for j in range(factor-1):
			test_data,test_label,training_data,training_label=k_cross_validation(X,label_list,j,k)
			print len(test_data),len(test_label),len(training_data),len(training_label)
			clf=LogisticRegression(max_iter=best_regularization_strength)
			clf.fit(training_data,training_label)
			score=clf.score(test_data,test_label)
			print listx[i],j,score
			if score>best:
				best=score
				itera=j
				factorval=listx[i]
	best_k_val=factorval
	print best,itera,factorval
	print len(reg_strength)
	print len(accr_val)

	#plt.figure(figsize=(12,8))


elif args.model_name == 'DecisionTreeClassifier':
	best_depth=0
	best_score=0
	accr_val=[]
	samples=[]
	for i in range(X_x/2):
		print "im in"
		clf=DecisionTreeClassifier(max_depth=i+1)
		samples.append(i+1)
		clf.fit(X,label_list)
		grid_score=clf.score(X,label_list)
		print i,grid_score
		accr_val.append(grid_score)
		if(grid_score>best_score):
			best_score=grid_score
			best_depth=i+1
			best_model=clf
		if(best_score==1):
			break
	plt.scatter(samples,accr_val)
	plt.savefig('Plots/result_decision{}'.format(t))
	plt.show()
	joblib.dump(best_model, 'Weights/DecisionTreeClassifier{}.pkl'.format(t))
	print "The best score obtained for changing hyperparameter ",best_depth,best_score
	listx=calc_factors(X_y)
	best=0
	itera=0
	factorval=0
	for i in range(len(listx)):
		k=listx[i]
		factor=X_y//k
		for j in range(factor-1):
			test_data,test_label,training_data,training_label=k_cross_validation(X,label_list,j,k)
			clf=DecisionTreeClassifier(max_depth=best_depth)
			clf.fit(training_data,training_label)
			score=clf.score(test_data,test_label)
			print listx[i],j,score
			if score>best:
				best=score
				itera=j
				factorval=listx[i]
	best_k_val=factorval
	print best,itera,factorval
	print len(samples)
	print len(accr_val)

elif args.model_name == 'selfgaussian':
	clf=GaussianNB()
	clf.fit(X, label_list)
	grid_score=clf.score(X,label_list)
	joblib.dump(clf, 'Weights/Self_GaussianNB{}.pkl'.format(t))
	print grid_score

	#	print "The best value obtained from grid search parameters for prior",best_prior_list,bestscore

elif args.model_name == 'selflogistic':
	clf=Models.LogisticRegression.LogisticRegression()
	clf.fit(X,label_list)
	print clf.predict(X)
	score=clf.score(X,label_list)
	joblib.dump(clf, 'Weights/Self_Logistic{}.pkl'.format(t))
	print score

elif args.model_name== 'selfdecission':
	clf=Models.DecisionTreeClassifier.DecisionTreeClassifier()
	label_list=np.array(label_list)
	clf.fit(X,label_list)
	score=clf.score(X,label_list)
	joblib.dump(clf, 'Weights/Self_DecisionTreeClassifier{}.pkl'.format(t))
	print score

else:
	raise Exception("Invald Model name")
