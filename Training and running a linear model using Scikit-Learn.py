#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:41:26 2024

@author: yuan
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors
# Load the data 
oecd_bli = pd.read_csv("/media/yuan/数据/研究生课件/AI与python/datasets/lifesat/oecd_bli_2015.csv",thousands=',')
gdp_per_capita = pd.read_csv("/media/yuan/数据/研究生课件/AI与python/datasets/lifesat/gdp_per_capita.csv",thousands=',',delimiter='\t',encoding='latin1',na_values='n/a')

#prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    #get the pandas dataframe of GDP per capita and Life satisfaction
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    return full_country_stats[["GDP per capita", 'Life satisfaction']]

country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter',x="GDP per capita",y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

model1 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
# Train the model
model.fit(X, y)
model1.fit(X, y)
# Make a prediction for Cyprus
X_new = [[22587]]
print(model.predict(X_new)) 
print(model1.predict(X_new))


