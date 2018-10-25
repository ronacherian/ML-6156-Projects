# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv')
#X = dataset.iloc[:, [0,1,2,3,4,5,7,11,13,14,15,17,19,20,21,22,23,24]].values
#y = dataset.iloc[:, -1]

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
dataset = dataset.apply(LabelEncoder().fit_transform)
X = dataset.iloc[:10000, :-1].values
y = dataset.iloc[:10000, -1].values

## Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#X[:, 1] = labelencoder.fit_transform(X[:, 1])
#onehotencoder = OneHotEncoder(categorical_features = [1])
#X = onehotencoder.fit_transform(X).toarray()
#
## Avoiding the Dummy Variable Trap
#X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(C=1e5)
# we create an instance of Neighbours Classifier and fit the data.
regressor.fit(X_train,y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# create the sub models
estimators = []
model1 = LogisticRegression(C=1e5)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
#create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,y_train)
# Predicting the Test set results
y_pred = ensemble.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
#results = model_selection.cross_val_score(ensemble, X_train, y)
#print(results)
