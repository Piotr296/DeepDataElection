# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:12:17 2019

@author: Petronium
"""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from sklearn.externals import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# training data
dataset_train = pd.read_csv('Data/Dane_treningowe_2005_2007_2011_csv.csv')

x_train = dataset_train[['MalTys', 'BudzM', 'BezPro', 'WspFem', 'WspObc', 'AptOs', 'PowUzM', 'BibM', 'WspSko', 'PodTys', 'OsbKM', 'WspX', 'WspY']].values
y_train = dataset_train[['Frekwencja']].values

# test data
dataset_test = pd.read_csv('Data/Dane_testowe_2015_csv.csv')

x_test = dataset_test[['MalTys', 'BudzM', 'BezPro', 'WspFem', 'WspObc', 'AptOs', 'PowUzM', 'BibM', 'WspSko', 'PodTys', 'OsbKM', 'WspX', 'WspY']].values
y_test = dataset_test[['Frekwencja']].values

# MMS 
mms = MinMaxScaler()

x_scaled_train = mms.fit_transform(x_train)
y_scaled_train = mms.fit_transform(y_train)

x_scaled_test = mms.fit_transform(x_test)
y_scaled_test = mms.fit_transform(y_test)

# ANN prediction
model = load_model('Data/Model_ANN.h5')
y_prediction = model.predict(x_scaled_test)
mae_ANN = mean_absolute_error(y_scaled_test, y_prediction)

# RFR prediction
rfr = joblib.load('Data/Model_RFR.pkl')
y_rfr_pred = rfr.predict(x_scaled_test)
y_rfr_pred = y_rfr_pred.reshape(2477,1)
mae_rfr = mean_absolute_error(y_scaled_test, y_rfr_pred)

# RFR charts
plt.scatter(y_test[:250], y_rfr_pred[:250]*100, color='orange')
plt.xlabel('Real results %')
plt.ylabel('Predicted results %')
plt.title('Comparison of predicted and real results')

sns.distplot((y_test-y_rfr_pred*100),bins=50,color='orange', kde=False)
plt.title('Histogram of the difference between real and predicted attendance')
plt.xlabel('The difference in attendance %')
plt.ylabel('Number of communes')

# ANN charts
plt.scatter(y_test[:250], y_prediction[:250]*100, color='orange')
plt.xlabel('Real results %')
plt.ylabel('Predicted results %')
plt.title('Comparison of predicted and real results')

sns.distplot((y_test-y_prediction*100),bins=50,color='orange', kde=False)
plt.title('Histogram of the difference between real and predicted attendance')
plt.xlabel('The difference in attendance %')
plt.ylabel('Number of communes')
