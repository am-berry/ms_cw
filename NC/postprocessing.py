#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

test_vals = pd.read_csv('net_test_errors.csv')
train_vals = pd.read_csv('net_train_errors.csv')

vals = pd.concat([train_vals, test_vals], axis = 0)
