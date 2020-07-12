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

# Build a DNN with 2 hidden layers wih 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3
)

classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)
# steps: similar to epochs, but runs until a certain number of items
# have been looked at as opposed to how many times it goes through data

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))


features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid: False
    
    predict[feature] = [float(val)]

# predictions = classifier.predict(input_fn=lambda: input_fn(predict))
# for pred_dict in predictions:
#     class_id = pred_dict['class_ids'][0]
#     probability = pred_dict['probabilities'][class_id]

# clustering: finds clusters of like data points and tells you the location of those clusters
# with training data, you can pick how many clusters you want to find

# K-Means Algorithm:
# Step 1: Randomly pick K points to place K controls
# Step 2: Assign all of the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
# Step 3: Average all of the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.
# Step 4: Reassign every point once again to the closest centroid.
# Step 5: Repeat steps 3-4 until no point changes which centroid it belongs to.

# Hidden Markov Models actually deal with probability distributions
# A hidden markov model works with probabilities to predict future events or states.

# Hidden Markov Model Example: Predict the Weather

# States: Finite set of states. They can be something like "warm", "cold", "high", "low"
# Observations: Each state has a particular outcome/observation associated with it based on probability distribution.
# Example: On a hot day Tim has an 80% chance of being happy and 20% chance of being sad.
# Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state.
# Example: A cold day has a 30% chance of being followed by a hot day and 70% chance of being followed by another cold day.

# Markov Models are different than linear regression or classification because they use probability distributions to predict future events or states.