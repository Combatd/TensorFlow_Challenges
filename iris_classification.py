# Classification is the process of separating data points into different classes.
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import os
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constatns to help us later on

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
test_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
train_path = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)
test_path = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url),
                                           origin=test_dataset_url)
print("Local copy of the dataset file: {}".format(train_path))

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train.head() # lets have a look at our data

train_y = train.pop('Species')
test_y = test.pop('Species')
train.head() # the species column is gone

train.shape # we have 12 entries with 4 features

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))