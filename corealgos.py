from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

# import tensorflow.compat.v2.feature_column as fc

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

print(dftrain.head()) # print out pandas data frame

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

print(dftrain.head())
# print(dftrain.loc[0], y_train.loc[0])

dftrain.describe() # statistical analysis of our data