from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import os
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf

# Hidden Markov Models actually deal with probability distributions
# A hidden markov model works with probabilities to predict future events or states.

# Hidden Markov Model Example: Predict the Weather

# States: Finite set of states. They can be something like "warm", "cold", "high", "low"
# Observations: Each state has a particular outcome/observation associated with it based on probability distribution.
# Example: On a hot day Tim has an 80% chance of being happy and 20% chance of being sad.
# Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state.
# Example: A cold day has a 30% chance of being followed by a hot day and 70% chance of being followed by another cold day.

# Markov Models are different than linear regression or classification because they use probability distributions to predict future events or states.

tfd = tfp.distributions

# A simple weather model.

# Represent a cold day with 0 and a hot day with 1.
# Suppose the first day of a sequence has a 0.8 chance of being cold.
# We can model this using the categorical distribution:

initial_distribution = tfd.Categorical(probs=[0.8, 0.2])

# Suppose a cold day has a 30% chance of being followed by a hot day
# and a hot day has a 20% chance of being followed by a cold day.
# We can model this as:

transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],
                                                 [0.2, 0.8]])

# Suppose additionally that on each day the temperature is
# normally distributed with mean and standard deviation 0 and 5 on
# a cold day and mean and standard deviation 15 and 10 on a hot day.
# We can model this with:

observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# We can combine these distributions into a single week long
# hidden Markov model with:

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)