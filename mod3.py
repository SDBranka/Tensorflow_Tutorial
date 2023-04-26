from __future__ import absolute_import, division, print_function, unicode_literals

# import numpy as np                                   # optimize arrays
import pandas as pd                                  # data analytics
import matplotlib.pyplot as plt                      # data visualization
import tensorflow as tf                              # needed to create a linear regression model algo
import tensorflow.compat.v2.feature_column as fc     # 
from IPython.display import clear_output             # to enable clearing the output
# from six.moves import urllib                         # 


# x = [1, 2, 2.5, 3, 4]
# y = [1, 4, 7, 9, 15]
# plt.plot(x, y, "ro")
# plt.axis([0, 6, 0, 20])
# # plt.show()

# store data from the csv's
df_train = pd.read_csv("Data/titanic_train.csv")
df_eval = pd.read_csv("Data/titanic_eval.csv")
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


# # inspect the shape of the data
# shape_df = df_train.shape
# # print(shape_df)
# # # (627, 9)


# # display comparison of passengers by age
# df_train.age.hist(bins = 20)
# # plt.show()


# # display comparison of passengers by sex
# df_train.sex.value_counts().plot(kind="barh")
# # plt.show()


# # display comparison of passengers by class
# df_train["class"].value_counts().plot(kind = "barh")
# # plt.show()


# # Compare survival rates by sex
# pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# # plt.show()


# Feature Columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 
                        'parch', 'class', 'deck',
                        'embark_town', 'alone'
                        ]
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = df_train[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# print(feature_columns)
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

# print(df_train["sex"].unique())
# # ['male' 'female']

# print(df_train["embark_town"].unique())
# # ['Southampton' 'Cherbourg' 'Queenstown' 'unknown']


# especially for larger data sets feeding the data into the program in batches can 
# make it easier for the computer to handle

# an epoch is one stream of the entire dataset fed in a different order, ie if there are 10 epochs the model
# will see the data 10 times

# overfitting - if the computer is fed the same data too many times it can make a model too
# specific to the data set, the computer essentially memorizes the dataset and develops a 
# model that predicts poorly when fed other sets. This is why it is better to start with
# a lower number of epochs and adjust up from there if needed

# Because we feed the data in batches and multiple times we need to create an input 
# function. This will define how the dataset will be converted into batches at each
# epoch, how the data will be broken into epochs
def make_input_fn(data_df, label_df, num_epochs = 10, shuffle = True, batch_size = 32):
    def input_function():            #inner function, this will be returned
        # create tf.data.Dataset object with data and it's label
        ds = tf. data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            # randomize the order of the data
            ds = ds.shuffle(1000)
        # split the dataset into batches and repeat the process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

# prepare the datasets for the model
train_input_fn = make_input_fn(df_train, y_train)
eval_input_fn = make_input_fn(df_eval, y_eval, num_epochs=1, shuffle=False)


# creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# training the model
# train
linear_est.train(train_input_fn)
# get model metrics/stats by testing on testing data
result = linear_est.evaluate(eval_input_fn)

# clear the console output
clear_output()
# the result is a dict of stats about the model
print(result["accuracy"])
# 0.74242425
# this means at first pass the model is about 74% accurate
# this may change each run as the computer may interpret 
# the data differently each time it is shuffled

# # what is stored in result
# print(result)
# # {'accuracy': 0.7348485, 'accuracy_baseline': 0.625, 'auc': 0.82133454, 'auc_precision_recall': 0.7615293, 'average_loss': 0.58852434, 'lab
# # el/mean': 0.375, 'loss': 0.5903315, 'precision': 0.610687, 'prediction/mean': 0.52226925, 'recall': 0.8080808, 'global_step': 200}

# if we want to actually check the predictions from the model
# made into a list here so we can loop through it
result_dict = list(linear_est.predict(eval_input_fn))
# print(result_dict)
# # see /Resources/result_output.txt


# print(result_dict[0])
# # {'logits': array([-2.1910503], dtype=float32), 'logistic': array([0.10055706], dtype=float32), 'probabilities': array([0.899443  , 0.10055
# # 706], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_c
# # lasses': array([b'0', b'1'], dtype=object)}


# # check the survival probability of the first entry
# print(result_dict[0]["probabilities"])
# # [0.9670563  0.03294368]
# # so about 97% chance of death, 3% chance of survival

# # let's take a look at the stats of this person
# print(df_eval.loc[0])
# # sex                          male
# # age                          35.0
# # n_siblings_spouses              0
# # parch                           0
# # fare                         8.05
# # class                       Third
# # deck                      unknown
# # embark_town           Southampton
# # alone                           y
# # Name: 0, dtype: object

# # let's see if this person actually survived
# print(y_eval.loc[0])
# # 0
# # 0 means did not survive (as mapped earlier 0 death, 1 survived)


# # !figuring out conversion
# print(result_dict[0]["probabilities"][1])
# probs = pd.Series([pred["probabilities"][1]].astype(float) for pred in result_dict)
# probs.astype(float)
# print(probs)
# # 0       [0.07950622]
# # 1       [0.35783988]
# # 2        [0.7451134]
# # 3       [0.66329175]
# # 4       [0.27935258]
# #            ...
# # 259      [0.8176828]
# # 260    [0.083364256]
# # 261      [0.5624116]
# # 262     [0.19784631]
# # 263     [0.41237444]
# # Length: 264, dtype: object

# print(probs)
# print(type(probs[0]))
# print(probs[0][0])
# 0.07263894


# # to display the predictions for survival of every person in the dataset
# probs = pd.Series([pred["probabilities"][1]] for pred in result_dict)

# probs_num = pd.Series([p[0] for p in probs])
# # print(probs_num)
# probs_num.plot(kind='hist', bins=20, title='predicted probabilities')
# plt.show()




# Classification
# Where regression was used to predict a numeric value, classification is used 
# to seperate data points into classes of different labels
# https://www.tensorflow.org/tutorials/estimator/premade
























