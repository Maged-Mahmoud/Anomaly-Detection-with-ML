# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:27:52 2020

@author: Maged
"""
# LocalOutlierFactor algorithm

# Import lib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

# load the dataset

path = "data.xlsx"
data = pd.read_excel(path, header = None)
data = data.values
print(data.shape)

# identify outliers in the dataset

lof = LocalOutlierFactor(contamination=0.13)
yhat = lof.fit_predict(data)

# select all rows that are not outliers

mask = yhat != -1
new_data = data[mask, :]
print(new_data.shape)
df = pd.DataFrame(new_data)
df.to_excel("new from Local.xlsx")

# Visualize the data after removing outliers

ax = plt.axes()
ax.scatter(new_data[:,0:1],new_data[:,1:],alpha=0.5)
ax.set(ylabel = 'Gas production rate, MSCF/d',xlabel = 'Production time, day', title = 'LocalOutlierFactor algorithm')




