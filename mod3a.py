# Predict the species of flower based on
# it's features

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np                                   # optimize arrays
import pandas as pd                                  # data analytics
import matplotlib.pyplot as plt                      # data visualization
import tensorflow as tf                              # needed to create a linear regression model algo
import tensorflow.compat.v2.feature_column as fc     # 
from IPython.display import clear_output             # to enable clearing the output
from six.moves import urllib                         # 


# Lets define some constants to help us later on
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


# Use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe
train_path = tf.keras.utils.get_file(
    "iris_train.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)


# # inspect the first five entries of the training data
# print(train.head())
# #    SepalLength  SepalWidth  PetalLength  PetalWidth  Species
# # 0          6.4         2.8          5.6         2.2        2
# # 1          5.0         2.3          3.3         1.0        1
# # 2          4.9         2.5          4.5         1.7        2
# # 3          4.9         3.1          1.5         0.1        0
# # 4          5.7         3.8          1.7         0.3        0


# pull our label to be predicted from the dfs
train_y = train.pop("Species")
test_y = test.pop("Species")
# # reinspect the df
# print(train.head())
# #    SepalLength  SepalWidth  PetalLength  PetalWidth
# # 0          6.4         2.8          5.6         2.2
# # 1          5.0         2.3          3.3         1.0
# # 2          4.9         2.5          4.5         1.7
# # 3          4.9         3.1          1.5         0.1
# # 4          5.7         3.8          1.7         0.3
# print(train.shape)
# # (120, 4)           # 120 entries with 4 columns


# create the input function
def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)


# print(train.keys())
# # Index(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'], dtype='object')


my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


# in mod 3 we predicted a numerical value and so we used linear regression
# as here we are attempting to predict a category we are using classification
# here we will build a Deep Neural Network Classifier, favored in this case over 
# a Linear Classifier as there might not be any linear correspondence in this data 
# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                        # Two hidden layers of 30 and 10 nodes respectively.
                                        # the number of hidden neurons is an arbitrary number and many experiments and tests are usually done to determine the best choice for these values
                                        hidden_units=[30, 10],
                                        # The model must choose between 3 classes.
                                        n_classes=3
                                        )


# train the model
# Using a lambda to avoid creating an inner function previously
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True),
                steps=5000
                )

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

# print(eval_result)
# {'accuracy': 0.93333334, 'average_loss': 0.47199103, 'loss': 0.47199103, 'global_step': 5000}


# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
# # Test set accuracy: 0.533

# Use trained model to make predictions
def input_fn(features, batch_size=256):
    # Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid: 
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" ({:.1f}%)'.format(
        SPECIES[class_id], 100 * probability))















