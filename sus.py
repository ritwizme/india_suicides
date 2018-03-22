# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:28:02 2018

@author: techwiz
"""

#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data cleaning and handling missing values
dataset = pd.read_csv("sus.csv")
dataset = dataset[dataset.Total != 0]
#Preparing features for Sucidal Analysis
#1.Andhra Pradesh
X = dataset.iloc[6712:13502,3:4].values
y = dataset.iloc[6712:13502,[6]].values

# Feature Scaling and Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
"""
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
#Removing Dummy Variable Trap
X = X[:,1:]
"""
#Creating Training and Test Sets
from sklearn.cross_validation import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
""""
#Fitting the model to Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
"""

#Using  Ensemble Learning approach to get better results
#Fitting model to RandomForestRegression
from sklearn.ensemble import RandomForestRegressor
regressor_RF = RandomForestRegressor()
regressor_RF.fit(X_train,y_train)
"""
# Predicting the Test set results
y_pred = regressor.predict(X_test)
#y_pred_rf = regressor_RF.predict(X_test)



# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Suicidal Analysis: Andhra Pradesh (Reason)')
plt.xlabel('Reasons')
plt.ylabel('No. of Suicides')
plt.show()

#Result analysis from viualisation
labelencoder_X.inverse_transform(40)
labelencoder_X.inverse_transform(39)
labelencoder_X.inverse_transform(38)
labelencoder_X.inverse_transform(45)
labelencoder_X.inverse_transform(43)