#!/usr/bin/python


#####decision tree classifier
##### Author: Priyanshu Jain

#import tree from sklearn
from sklearn import tree
def classify(feature,train):

    #create classifier
    clf = tree.DecisionTreeClassifier()
    
    #fit the training data
    clf.fit(feature,train)
    
    return clf
