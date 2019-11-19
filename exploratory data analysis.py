# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# training data
dataset_train = pd.read_csv('Data/Dane_treningowe_2005_2007_2011_csv.csv')

# joinplots
sns.jointplot(x='Frekwencja',y='BezPro',data=dataset_train, color='orange', kind='kde')
sns.jointplot(x='Frekwencja',y='MalTys',data=dataset_train, color='orange', kind='kde')
sns.jointplot(x='Frekwencja',y='OsbKM',data=dataset_train, color='orange', kind='kde')

# pairplot
sns.pairplot(dataset_train)

# correlation
x_axis_labels = ['Marriages / 1000 citizens', 'Commune income per capita', 'Registered unemployment', 'Feminisation coefficient', 'Age dependency ratio', 'Population / 1 pharmacy', 'Usable floor area / 1 inhabitant', 'Population / 1 library', ' Enrolment ratio', 'Economic entities / 10,000 citizens', 'Population density', 'X-coordinate', 'Y-coordinate', 'Attendance']
plt.figure(figsize=(8,8))
voter_data = dataset_train[['MalTys', 'BudzM', 'BezPro', 'WspFem', 'WspObc', 'AptOs', 'PowUzM', 'BibM', 'WspSko', 'PodTys', 'OsbKM', 'WspX', 'WspY', 'Frekwencja']]
sns.clustermap(voter_data.corr(), cmap="YlGnBu", square=True, center=0, linewidths=0.5, xticklabels=x_axis_labels, yticklabels=x_axis_labels )
plt.title('Autocorrelation chart for training data')