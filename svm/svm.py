import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()


########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(C=10000000,kernel="rbf",gamma=100)


#### now your job is to fit the classifier
clf.fit(features_train,labels_train)
#### using the training features/labels, and to
#### make a set of predictions on the test data



#### store your predictions in a list named pred
pred = clf.predict(features_test)

prettyPicture(clf,features_test,labels_test)
plt.show()


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
print "accuracy:",acc

