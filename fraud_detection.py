#!/usr/bin/python
# Author:  Abhinav
#

# Imports

import numpy as np
import pandas as panda
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Load data using panda

data = panda.read_csv('./creditcard.csv')

data.Class.value_counts()

# Create Training and Test data
X = data.ix[:,'V1':'Amount'].as_matrix()
y = data.Class.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)


logistic = linear_model.LogisticRegression(class_weight='balanced')
logistic.fit(X_train, y_train)

predictions = logistic.predict(X_test)

print "the accuracy of this model is :"