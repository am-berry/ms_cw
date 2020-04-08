#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dat = pd.read_csv('data.csv')

X = dat.drop('diagnosis', axis = 1)
y = dat['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

train = pd.concat([X_train, y_train], axis = 1)
test = pd.concat([X_test, y_test], axis = 1)

for df in [train, test]:
  df.reset_index()
  df.drop('Unnamed: 32', axis = 1, inplace=True)

train.to_csv('train_data.csv')
test.to_csv('test_data.csv')
