# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from keras.models import Model, Input
from keras.optimizers import SGD
from keras.layers import Dense

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
x_scaled_test = mms.transform(x_test)

y_scaled_train = mms.fit_transform(y_train)
y_scaled_test = mms.transform(y_test)

# ANN architecture
inputs = Input(shape=[13])
layer = Dense(512, activation='relu')(inputs)
layer = Dense(256, activation='relu')(layer)
layer = Dense(128, activation='relu')(layer)
layer = Dense(64, activation='relu')(layer)
layer = Dense(32, activation='relu')(layer)
layer = Dense(16, activation='relu')(layer)
layer = Dense(8, activation='relu')(layer)
layer = Dense(4, activation='relu')(layer)
output = Dense(1, activation='sigmoid')(layer)

model = Model(inputs=inputs, outputs=output)
model.summary()

# ANN parameters 
epochs = 1000
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.9

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['mae'])

model.fit(x=x_scaled_train,
          y=y_scaled_train,
          batch_size=30,
          epochs=epochs,
          validation_data=[x_scaled_test, y_scaled_test])

model.save('Model_ANN.h5')

# ANN prediction
y_prediction = model.predict(x_scaled_test)
mae_ANN = mean_absolute_error(y_scaled_test, y_prediction)

# RFR 
rfr = RandomForestRegressor()
rfr.fit(x_scaled_train, y_scaled_train)
joblib.dump(rfr, 'Model_RFR.pkl') 

# RFR prediction
y_rfr_pred = rfr.predict(x_scaled_test)
y_rfr_pred = y_rfr_pred.reshape(2477,1)
mae_rfr = mean_absolute_error(y_scaled_test, y_rfr_pred)

# Save
dataset_test['ANN_pred'] = y_prediction*100
dataset_test.to_excel("Results2015-ANN.xlsx", sheet_name='Tabel')

dataset_test['RFR_pred'] = y_rfr_pred*100
dataset_test.to_excel("Results2015-RFR.xlsx", sheet_name='Tabel')