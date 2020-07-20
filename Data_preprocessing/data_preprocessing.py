#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:24:02 2020

@author: beinexconsulting
"""

#Library Part

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Import DataSet 

dataset = pd.read_csv('Data.csv')
matrix_feature = dataset.iloc[ : , :-1 ].values
dependent_variable = dataset.iloc[ : , -1].values

# Take Care Missing Data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',)
imputer.fit(matrix_feature[:,1:3])
matrix_feature[:,1:3] = imputer.transform(matrix_feature[:,1:3])
print(matrix_feature)
