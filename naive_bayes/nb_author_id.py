#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
print "Type of features_train is:", type(features_train)
#Time training
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

#Time of a single prediction
t0 = time()
print "Prediction:",clf.predict(features_train[0])
print "prediction time:", round(time()-t0, 3), "s"

accuracy = clf.score(features_test,labels_test)
print "Accuracy is",accuracy

probability = clf.predict_proba(features_test)
print "Probability is", probability

print "Example datasets....."
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
from sklearn import svm
clf2 = svm.SVC(gamma = 0.001, C=100.)
clf2.fit(digits.data[:-1], digits.target[:-1])
print clf2.predict(digits.data[-1:])


#########################################################


