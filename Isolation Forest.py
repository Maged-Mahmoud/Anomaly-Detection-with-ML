# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:09:38 2020

@author: Maged
"""
# IsolationForest algorithm

# Import lib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# load the dataset

path = "data.xlsx"
data = pd.read_excel(path, header = None)
data = data.values
print(data.shape)
# identify outliers in the dataset

iso = IsolationForest(random_state=44, n_estimators= 100, contamination=0.15 )
yhat = iso.fit_predict(data)
# print(yhat)

# select all rows that are not outliers

mask = yhat != -1
new_data = data[mask, :]
print(new_data.shape)
# print(iso.score_samples(new_data))
df = pd.DataFrame(new_data)
df.to_excel("new from Isolation.xlsx")

# Visualize the data after removing outliers

ax = plt.axes()
ax.scatter(new_data[:,0:1],new_data[:,1:],alpha=0.8)
ax.set(ylabel = 'Gas production rate, MSCF/d',xlabel = 'Production time, day', title = 'IsolationForest algorithm')