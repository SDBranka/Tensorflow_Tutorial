from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np                                   # optimize arrays
import pandas as pd                                  # data analytics
import matplotlib.pyplot as plt                      # data visualization
from IPython.display import clear_output             # to enable clearing the output
from six.moves import urllib                         # 
import tensorflow as tf                              # needed to create a linear regression model algo
import tensorflow.compat.v2.feature_column as fc     # 


# x = [1, 2, 2.5, 3, 4]
# y = [1, 4, 7, 9, 15]
# plt.plot(x, y, "ro")
# plt.axis([0, 6, 0, 20])
# # plt.show()

# store data from the csv's
df_train = pd.read_csv("Data/train.csv")
df_eval = pd.read_csv("Data/eval.csv")
# print(df_train)
# #      survived     sex   age  ...     deck  embark_town  alone
# # 0           0    male  22.0  ...  unknown  Southampton      n
# # 1           1  female  38.0  ...        C    Cherbourg      n
# # 2           1  female  26.0  ...  unknown  Southampton      y
# # 3           1  female  35.0  ...        C  Southampton      n
# # 4           0    male  28.0  ...  unknown   Queenstown      y
# # ..        ...     ...   ...  ...      ...          ...    ...
# # 622         0    male  28.0  ...  unknown  Southampton      y
# # 623         0    male  25.0  ...  unknown  Southampton      y
# # 624         1  female  19.0  ...        B  Southampton      y
# # 625         0  female  28.0  ...  unknown  Southampton      n
# # 626         0    male  32.0  ...  unknown   Queenstown      y
# print(df_train.head())
# #    survived     sex   age  n_siblings_spouses  parch     fare  class     deck  embark_town alone
# # 0         0    male  22.0                   1      0   7.2500  Third  unknown  Southampton     n
# # 1         1  female  38.0                   1      0  71.2833  First        C    Cherbourg     n
# # 2         1  female  26.0                   0      0   7.9250  Third  unknown  Southampton     y
# # 3         1  female  35.0                   1      0  53.1000  First        C  Southampton     n
# # 4         0    male  28.0                   0      0   8.4583  Third  unknown   Queenstown     y


# store "survived" column from each df as a variable
# it is beneficial to store the data we intend to classify separately
# from the data we intend to base our predictions upon(input data). As we 
# intend to predict the survival rate we remove this column from the df 
# and store it as a variable
y_train = df_train.pop("survived")
y_eval = df_eval.pop("survived")
# print(y_train)
# # 0      0
# # 1      1
# # 2      1
# # 3      1
# # 4      0
# #       ..
# # 622    0
# # 623    0
# # 624    1
# # 625    0
# # 626    0


# inspect the first five items of the df
# after removing the "survived" column 
first_five_rows = df_train.head()
# print(first_five_rows)
# #       sex   age  n_siblings_spouses  parch  ...  class     deck  embark_town alone
# # 0    male  22.0                   1      0  ...  Third  unknown  Southampton  n
# # 1  female  38.0                   1      0  ...  First        C    Cherbourg  n
# # 2  female  26.0                   0      0  ...  Third  unknown  Southampton  y
# # 3  female  35.0                   1      0  ...  First        C  Southampton  n
# # 4    male  28.0                   0      0  ...  Third  unknown   Queenstown  y

# # [5 rows x 9 columns]

# y_train_first_five = y_train.head()

# # inspect values at specific locations (in this case row 0)
# print(df_train.loc[0])
# # sex                          male
# # age                          22.0
# # n_siblings_spouses              1
# # parch                           0
# # fare                         7.25
# # class                       Third
# # deck                      unknown
# # embark_town           Southampton
# # alone                           n
# # Name: 0, dtype: object
# print(y_train.loc[0])
# # 0
# # inspect a column of the df
# print(df_train["age"])
# # 0      22.0
# # 1      38.0
# # 2      26.0
# # 3      35.0
# # 4      28.0
# #        ...
# # 622    28.0
# # 623    25.0
# # 624    19.0
# # 625    28.0
# # 626    32.0
# # Name: age, Length: 627, dtype: float64


# more statistical analysis of data. gives some overall 
# information about the df 
describe_data = df_train.describe()
# print(describe_data)
# #               age  n_siblings_spouses       parch        fare
# # count  627.000000          627.000000  627.000000  627.000000
# # mean    29.631308            0.545455    0.379585   34.385399
# # std     12.511818            1.151090    0.792999   54.597730
# # min      0.750000            0.000000    0.000000    0.000000
# # 25%     23.000000            0.000000    0.000000    7.895800
# # 50%     28.000000            0.000000    0.000000   15.045800
# # 75%     35.000000            1.000000    0.000000   31.387500
# # max     80.000000            8.000000    5.000000  512.329200


# inspect the shape of the data
shape_df = df_train.shape
# print(shape_df)
# # (627, 9)


df_train.age.hist(bins = 20)
# plt.show()


df_train.sex.value_counts().plot(kind="barh")
# plt.show()


df_train["class"].value_counts().plot(kind = "barh")
# plt.show()


pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()


# Feature Columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 
                        'parch', 'class', 'deck',
                        'embark_town', 'alone'
                        ]
NUMERIC_COLUMNS = ['age', 'fare']

# feature_columns = []
# for feature_name in CATEGORICAL_COLUMNS:
#     vocabulary = df_train[feature_name].unique()  # gets a list of all unique values from given feature column
#     feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

# for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)
# WARNING:tensorflow:From C:\Users\xxxxxx\Desktop\Programming\Projects\Machine_Lear
# ning\Tensorflow_Tutorial\mod3.py:120: categorical_column_with_vocabulary_list (f
# rom tensorflow.python.feature_column.feature_column_v2) is deprecated and will b
# e removed in a future version.
# Instructions for updating:
# Use Keras preprocessing layers instead, either directly or via the `tf.keras.uti
# ls.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equival
# ent in `tf.keras.layers` for feature preprocessing when training a Keras model.
# WARNING:tensorflow:From C:\Users\xxxxxx\Desktop\Programming\Projects\Machine_Lear
# ning\Tensorflow_Tutorial\mod3.py:123: numeric_column (from tensorflow.python.fea
# ture_column.feature_column_v2) is deprecated and will be removed in a future ver
# sion.
# Instructions for updating:
# Use Keras preprocessing layers instead, either directly or via the `tf.keras.uti
# ls.FeatureSpace` utility. Each of `tf.feature_column.*` has a functional equival
# ent in `tf.keras.layers` for feature preprocessing when training a Keras model.

# [VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'),
# dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategorical
# Column(key='n_siblings_spouses', vocabulary_list=(1, 0, 3, 4, 2, 5, 8), dtype=tf
# .int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(ke
# y='parch', vocabulary_list=(0, 1, 2, 5, 3, 4), dtype=tf.int64, default_value=-1,
#  num_oov_buckets=0), VocabularyListCategoricalColumn(key='class', vocabulary_lis
# t=('Third', 'First', 'Second'), dtype=tf.string, default_value=-1, num_oov_bucke
# ts=0), VocabularyListCategoricalColumn(key='deck', vocabulary_list=('unknown', '
# C', 'G', 'A', 'B', 'D', 'F', 'E'), dtype=tf.string, default_value=-1, num_oov_bu
# ckets=0), VocabularyListCategoricalColumn(key='embark_town', vocabulary_list=('S
# outhampton', 'Cherbourg', 'Queenstown', 'unknown'), dtype=tf.string, default_val
# ue=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='alone', vocabula
# ry_list=('n', 'y'), dtype=tf.string, default_value=-1, num_oov_buckets=0), Numer
# icColumn(key='age', shape=(1,), default_value=None, dtype=tf.float32, normalizer
# _fn=None), NumericColumn(key='fare', shape=(1,), default_value=None, dtype=tf.fl
# oat32, normalizer_fn=None)]











