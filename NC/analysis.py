#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
data.drop(['Unnamed: 32', 'id'], axis = 1, inplace=True)
corr = data.corr()
ax = sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, cmap = sns.diverging_palette(20, 220, n = 200), square = True, xticklabels = True, yticklabels = True)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')
#plt.show()

data.drop(['perimeter_mean', 'area_mean', 'concavity_mean','concave points_mean', 'perimeter_se', 'area_se','concavity_se', 'fractal_dimension_se','radius_worst','texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst','concave points_worst', 'fractal_dimension_worst'], axis = 1, inplace=True)

print(data.head())
