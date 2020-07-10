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

# epoch: the number of items the model will see the same data
# For massive datasets, we need to load it in batches
# overfitting: we pass too much data to the model and it memorizes the data points
# We should start with a lower amount of epochs and incrementally increase batches

# Input Function: the way we define data is gonna be broken into epochs
# TensorFlow model data comes in as a tf.data.Dataset object
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create tf.data/Dataset object with data
        if shuffle:
            ds = ds.shuffle(1000) # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process for number of epochs
        return ds # return a batch of the dataset
    return input_function # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# estimators: basic implementations of algorithms
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)