# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:16:24 2025

@author: salih
"""

import pandas as pd

veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,-1].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred2 = knn.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

from sklearn.svm import SVC
svc = SVC(kernel ='linear')
svc.fit(X_train, y_train)

y_pred3 = svc.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)