#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""   

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import matplotlib.pyplot as plt
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

#########################################################

clf = svm.SVC(C=10000,kernel = "rbf")
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
initial_time = time()
clf.fit(features_train, labels_train)
print "Training time is:", time()-initial_time
print "Accuracy is:", clf.score(features_test,labels_test) 
i = 0
count = 0
for feature in features_test:
	if clf.predict(feature) == 1:
		count+=1
print count
