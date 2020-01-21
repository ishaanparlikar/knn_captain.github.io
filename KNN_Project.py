#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 02:42:02 2020

@author: captain

"""


from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("KNN_Project_Data")


scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()
X = scaled_features
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)

error_rate = []
for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), error_rate, marker='o', color='red', visible='True')
plt.xlabel('k')
plt.ylabel('Error Rate')
plt.show()

print('\n')

viz = ClassificationReport(KNeighborsClassifier(n_neighbors=2))
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
pred1 = knn.predict(X_test)
cf1 = confusion_matrix(y_test, pred1)
# print(cf1_m)
# print('\n')
cr1 = classification_report(y_test, pred1)
print(cr1)
viz1 = ClassificationReport(KNeighborsClassifier(n_neighbors=31))
viz1.fit(X_train, y_train)
viz1.score(X_test, y_test)
viz1.show()
knn = KNeighborsClassifier(n_neighbors=31)
knn.fit(X_train, y_train)
pred2 = knn.predict(X_test)
cf31 = confusion_matrix(y_test, pred2)
# print(cf31_m)

cr2 = classification_report(y_test, pred2)
print(cr2)
