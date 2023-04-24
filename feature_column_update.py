# https://www.tensorflow.org/guide/migrate/migrating_feature_columns


import tensorflow as tf
import tensorflow.compat.v1 as tf1
import math


# add a utility function for calling a feature column for demonstration
def call_feature_columns(feature_columns, inputs):
    # This is a convenient way to call a `feature_column` outside of an estimator
    # to display its output.
    feature_layer = tf1.keras.layers.DenseFeatures(feature_columns)
    return feature_layer(inputs)


# To use feature columns with an estimator, model inputs are always expected 
# to be a dictionary of tensors
input_dict = {
    'foo': tf.constant([1]),
    'bar': tf.constant([0]),
    'baz': tf.constant([-1])
}

# Each feature column needs to be created with a key to index into the source 
# data. The output of all feature columns is concatenated and used by the 
# estimator model
# columns = [
#     tf1.feature_column.numeric_column('foo'),
#     tf1.feature_column.numeric_column('bar'),
#     tf1.feature_column.numeric_column('baz'),
# ]
# call_feature_columns(columns, input_dict)

# WARNING:tensorflow:From /tmpfs/tmp/ipykernel_16900/3124623333.py:2: numeric_column 
# (from tensorflow.python.feature_column.feature_column_v2) is deprecated and will be 
# removed in a future version.
# Instructions for updating:
# Use Keras preprocessing layers instead, either directly or via the 
# `tf.keras.utils.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional 
# equivalent in `tf.keras.layers` for feature preprocessing when training a Keras model.
# <tf.Tensor: shape=(1, 3), dtype=float32, numpy=array([[ 0., -1.,  1.]], dtype=float32)>


# In Keras, model input is much more flexible. A tf.keras.Model can handle a 
# single tensor input, a list of tensor features, or a dictionary of tensor features. 
# You can handle dictionary input by passing a dictionary of tf.keras.Input on model 
# creation. Inputs will not be concatenated automatically, which allows them to be 
# used in much more flexible ways. They can be concatenated with tf.keras.layers.Concatenate

inputs = {
    'foo': tf.keras.Input(shape=()),
    'bar': tf.keras.Input(shape=()),
    'baz': tf.keras.Input(shape=()),
}
# # Inputs are typically transformed by preprocessing layers before concatenation.
outputs = tf.keras.layers.Concatenate()(inputs.values())
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model(input_dict)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([ 1.,  0., -1.], dtype=float32)>