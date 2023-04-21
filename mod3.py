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


# store "survived" column from each df as a variable
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
first_five_rows = df_train.head()
# print(first_five_rows)
# #       sex   age  n_siblings_spouses  parch  ...  class     deck  embark_town alo
# # ne
# # 0    male  22.0                   1      0  ...  Third  unknown  Southampton
# #  n
# # 1  female  38.0                   1      0  ...  First        C    Cherbourg
# #  n
# # 2  female  26.0                   0      0  ...  Third  unknown  Southampton
# #  y
# # 3  female  35.0                   1      0  ...  First        C  Southampton
# #  n
# # 4    male  28.0                   0      0  ...  Third  unknown   Queenstown
# #  y

# # [5 rows x 9 columns]

y_train_first_five = y_train.head()


# more statistical analysis of data 
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













