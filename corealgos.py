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

# Categorical data: Any data that is not numeric
# We encode data like gender as numerical data - 0 and 1
# 1st, 2nd, 3rd class could be 0, 1, 2....

# dftrain["sex"].unique() # get unique values of sex - M, F

CATEGORICAL_COLUMNS = ['sex' 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# need to make feature_columns for linear regression
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # get a list of all unique values from ggiven feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)