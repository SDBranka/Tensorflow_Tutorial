# Attempt to predict the weather on a given day


# Hidden Markov Models
# Deals with probability distributions
# "The Hidden Markov Model is a finite set of states, each of 
# which is associated with a (generally multidimensional) 
# probability distribution []. Transitions among the states are 
# governed by a set of probabilities called transition 
# probabilities."

# A hidden markov model works with probabilities to predict
# future events or states. In this section we will learn how to 
# create a hidden markov model that can predict the weather.

# Data we use when we work with a hidden markov model.
# In the previous sections we worked with large datasets of 100's
# of different entries. For a markov model we are only interested
# in probability distributions that have to do with states.
# We can find these probabilities from large datasets or may 
# already have these values

# States: In each markov model we have a finite set of states. 
# These states could be something like "warm" and "cold" or 
# "high" and "low" or even "red", "green" and "blue". These 
# states are "hidden" within the model, which means we do not 
# direcly observe them.

# Observations: Each state has a particular outcome or observation
# associated with it based on a probability distribution. An 
# example of this is the following: On a hot day Tim has a 80% 
# chance of being happy and a 20% chance of being sad.

# Transitions: Each state will have a probability defining the 
# likelyhood of transitioning to a different state. An example is
# the following: a cold day has a 30% chance of being followed by
# a hot day and a 70% chance of being follwed by another cold day.

# To create a hidden markov model we need:
# States
# Observation Distribution
# Transition Distribution

# Talking about data, in the first two examples we needed large
# datasets to train and build the model; with hidden markov models
# all we need is constant values for transition distributions and 
# observation distributions


# transition probability
# (s1) on a hot day there is a 20% chance of it transitioning to a cold day
# and a 80% chance of it transitioning to another hot day
# (s2) on a cold day there is a 30% chance of it transitioning to a hot day
# and a 70% chance of it transitioning to another cold day
# observation distribution
# on a hot day the temp will be between 15-25 degrees Celsius with
# an average temp of 20
# on a cold day the temp will be between -5-15 degrees Celsius
# with an average temp of 5

# Cold Days are encoded by 0 and hot days are encoded by 1
# The first day in our sequence has an 80% chance of being cold.
# A cold day has a 30% chance of being followed by a hot day.
# A hot day has a 20% chance of being followed by a cold day
# On each day the temperature is normally distributed with mean 
# and standard deviation 0 and 5 on a cold day and mean and 
# standard deviation 15 and 10 on a hot day.
# In this example, on a hot day the average temp is 15 and ranges 5-25







import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf


# making a shortcut for later on
tfd = tfp.distributions 


# based on our initial observation that on a hot day there's a 20%
# of the next day being cold and an 80% chance of the next day being hot
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])  
# a cold day (state 0) has a 70% chance of converting to another cold day and a 30% chance of converting to a hot day
# a hot day (state 1) has a 80% chance of converting to another hot day and a 20% chance of converting to a cold day
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],    # probs for transition cold day
                                                [0.2, 0.8]]     # probs for transition hot day
                                                )
# the loc argument represents the mean and the scale is 
# the standard devitation
# forcing the values to be floats in this instance
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])


# create the model
# The number of steps represents the number of days that we 
# would like to predict information for
model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)


# predict the average temp for each day
mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:  
    print(mean.numpy())








# Traceback (most recent call last):
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\mod3b.py", line 76, in <module>
#     import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\__init__.p
# y", line 75, in <module>
#     from tensorflow_probability.python import *  # pylint: disable=wildcard-import
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\python\__i
# nit__.py", line 24, in <module>
#     from tensorflow_probability.python import edward2
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\python\edw
# ard2\__init__.py", line 32, in <module>
#     from tensorflow_probability.python.experimental.edward2.generated_random_variables import *
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\python\exp
# erimental\__init__.py", line 34, in <module>
#     from tensorflow_probability.python.experimental import auto_batching
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\python\exp
# erimental\auto_batching\__init__.py", line 24, in <module>
#     from tensorflow_probability.python.experimental.auto_batching import frontend
#   File "C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Tensorflow_Tutorial\tfEnv\lib\site-packages\tensorflow_probability\python\exp
# erimental\auto_batching\frontend.py", line 44, in <module>
#     from tensorflow.python.autograph.core import naming
# ImportError: cannot import name 'naming' from 'tensorflow.python.autograph.core' (C:\Users\sdbra\Desktop\Programming\Projects\Machine_Learning\Ten
# sorflow_Tutorial\tfEnv\lib\site-packages\tensorflow\python\autograph\core\__init__.py)


