#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:24:02 2020

@author: Anoop
"""

#Library Part

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Import DataSet 

dataset = pd.read_csv('Data.csv')
matrix_feature = dataset.iloc[ : , :-1 ].values
dependent_variable = dataset.iloc[ : , -1].values

# Take Care Missing Data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',)
imputer.fit(matrix_feature[:,1:3])
matrix_feature[:,1:3] = imputer.transform(matrix_feature[:,1:3])

# Encoding Categorical Data
ct = ColumnTransformer(transformers= [('encoder',OneHotEncoder(),[0])], remainder="passthrough")
matrix_feature =np.array( ct.fit_transform(matrix_feature))

label = LabelEncoder()
dependent_variable = label.fit_transform(dependent_variable)

# Splitting the data set into Training Set and Test Set
matrix_feature_train,matrix_feature_test , \
  dependent_variable_train, dependent_variable_test = train_test_split(matrix_feature,dependent_variable,test_size = 0.2, random_state = 1)
  

# Feature Scaling
  
scale = StandardScaler()
matrix_feature_train[:,3:] = scale.fit_transform(matrix_feature_train[:,3:])
matrix_feature_test[:,3:] = scale.fit_transform(matrix_feature_test[:,3:])

print(matrix_feature_train)

print(("**************************"))

print(matrix_feature_test)