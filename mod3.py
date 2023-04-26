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


# inspect the shape of the data
shape_df = df_train.shape
# print(shape_df)
# # (627, 9)


# display comparison of passengers by age
df_train.age.hist(bins = 20)
# plt.show()


# display comparison of passengers by sex
df_train.sex.value_counts().plot(kind="barh")
# plt.show()


# display comparison of passengers by class
df_train["class"].value_counts().plot(kind = "barh")
# plt.show()


# Compare survival rates by sex
pd.concat([df_train, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# plt.show()


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
result = list(linear_est.predict(eval_input_fn))
print(result)
# [{'logits': array([-3.2225733], dtype=float32), 'logistic': array([0.03832503], dtype=float32), 'probabilities': array([0.961675  , 0.0383
# 2503], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_
# classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.7973497], dtype=float32), 'logistic': array([0.14217399], dtype=float32
# ), 'probabilities': array([0.85782593, 0.14217399], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=o
# bject), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.08219753], dtype=float32),
#  'logistic': array([0.47946218], dtype=float32), 'probabilities': array([0.5205378 , 0.47946215], dtype=float32), 'class_ids': array([0],
# dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {
# 'logits': array([-0.4458251], dtype=float32), 'logistic': array([0.39035383], dtype=float32), 'probabilities': array([0.60964614, 0.390353
# 83], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cl
# asses': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.7108501], dtype=float32), 'logistic': array([0.15305349], dtype=float32),
#  'probabilities': array([0.84694654, 0.15305349], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=obj
# ect), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.6456787], dtype=float32), 'lo
# gistic': array([0.656036], dtype=float32), 'probabilities': array([0.34396398, 0.656036  ], dtype=float32), 'class_ids': array([1], dtype=
# int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logit
# s': array([-0.18084954], dtype=float32), 'logistic': array([0.45491046], dtype=float32), 'probabilities': array([0.54508954, 0.45491043],
# dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes
# ': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6778107], dtype=float32), 'logistic': array([0.06429546], dtype=float32), 'pro
# babilities': array([0.9357045 , 0.06429546], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object),
#  'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.1114224], dtype=float32), 'logist
# ic': array([0.47217318], dtype=float32), 'probabilities': array([0.5278268 , 0.47217315], dtype=float32), 'class_ids': array([0], dtype=in
# t64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits'
# : array([0.48978913], dtype=float32), 'logistic': array([0.62005675], dtype=float32), 'probabilities': array([0.37994325, 0.62005675], dty
# pe=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes':
# array([b'0', b'1'], dtype=object)}, {'logits': array([0.13937499], dtype=float32), 'logistic': array([0.5347874], dtype=float32), 'probabi
# lities': array([0.46521258, 0.5347875 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'al
# l_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6782162], dtype=float32), 'logistic':
#  array([0.06427108], dtype=float32), 'probabilities': array([0.93572897, 0.06427108], dtype=float32), 'class_ids': array([0], dtype=int64)
# , 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': ar
# ray([1.3521214], dtype=float32), 'logistic': array([0.7944762], dtype=float32), 'probabilities': array([0.20552377, 0.7944763 ], dtype=flo
# at32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array(
# [b'0', b'1'], dtype=object)}, {'logits': array([-1.2125809], dtype=float32), 'logistic': array([0.22924471], dtype=float32), 'probabilitie
# s': array([0.7707553, 0.2292447], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class
# _ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.024449], dtype=float32), 'logistic': array([0
# .7358383], dtype=float32), 'probabilities': array([0.2641617, 0.7358383], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes':
#  array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.4231
# 887], dtype=float32), 'logistic': array([0.39575398], dtype=float32), 'probabilities': array([0.604246  , 0.39575398], dtype=float32), 'cl
# ass_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1
# '], dtype=object)}, {'logits': array([-2.244729], dtype=float32), 'logistic': array([0.0958051], dtype=float32), 'probabilities': array([0
# .90419495, 0.09580511], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': arr
# ay([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.023893], dtype=float32), 'logistic': array([0.73573023]
# , dtype=float32), 'probabilities': array([0.26426977, 0.73573023], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array(
# [b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.3095353], d
# type=float32), 'logistic': array([0.21256462], dtype=float32), 'probabilities': array([0.78743535, 0.2125646 ], dtype=float32), 'class_ids
# ': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dty
# pe=object)}, {'logits': array([-2.9891033], dtype=float32), 'logistic': array([0.04792058], dtype=float32), 'probabilities': array([0.9520
# 794 , 0.04792058], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0
# , 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6392167], dtype=float32), 'logistic': array([0.06665675], d
# type=float32), 'probabilities': array([0.93334323, 0.06665675], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'
# 0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.0187454], dtyp
# e=float32), 'logistic': array([0.04658617], dtype=float32), 'probabilities': array([0.95341384, 0.04658617], dtype=float32), 'class_ids':
# array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=
# object)}, {'logits': array([0.06176378], dtype=float32), 'logistic': array([0.515436], dtype=float32), 'probabilities': array([0.48456398,
#  0.51543605], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1])
# , 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.3395581], dtype=float32), 'logistic': array([0.03423877], dtype=
# float32), 'probabilities': array([0.96576124, 0.03423877], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'],
# dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.12343396], dtype=flo
# at32), 'logistic': array([0.53081936], dtype=float32), 'probabilities': array([0.4691806 , 0.53081936], dtype=float32), 'class_ids': array
# ([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=objec
# t)}, {'logits': array([-0.02024625], dtype=float32), 'logistic': array([0.49493858], dtype=float32), 'probabilities': array([0.5050614 , 0
# .49493858], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]),
# 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.076481], dtype=float32), 'logistic': array([0.11140384], dtype=flo
# at32), 'probabilities': array([0.8885962 , 0.11140385], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dty
# pe=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1268878], dtype=float3
# 2), 'logistic': array([0.1065108], dtype=float32), 'probabilities': array([0.8934892, 0.1065108], dtype=float32), 'class_ids': array([0],
# dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {
# 'logits': array([-1.7051783], dtype=float32), 'logistic': array([0.15379016], dtype=float32), 'probabilities': array([0.84620976, 0.153790
# 15], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cl
# asses': array([b'0', b'1'], dtype=object)}, {'logits': array([1.0454084], dtype=float32), 'logistic': array([0.7398922], dtype=float32), '
# probabilities': array([0.26010782, 0.73989224], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=objec
# t), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.54698], dtype=float32), 'logis
# tic': array([0.07262964], dtype=float32), 'probabilities': array([0.9273703 , 0.07262964], dtype=float32), 'class_ids': array([0], dtype=i
# nt64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits
# ': array([0.60133016], dtype=float32), 'logistic': array([0.64596057], dtype=float32), 'probabilities': array([0.35403943, 0.64596057], dt
# ype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes':
#  array([b'0', b'1'], dtype=object)}, {'logits': array([-2.0235672], dtype=float32), 'logistic': array([0.11675064], dtype=float32), 'proba
# bilities': array([0.8832494 , 0.11675065], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), '
# all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.02243792], dtype=float32), 'logisti
# c': array([0.49439076], dtype=float32), 'probabilities': array([0.5056093, 0.4943908], dtype=float32), 'class_ids': array([0], dtype=int64
# ), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': a
# rray([-1.5849181], dtype=float32), 'logistic': array([0.17010008], dtype=float32), 'probabilities': array([0.82989985, 0.17010008], dtype=
# float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': arr
# ay([b'0', b'1'], dtype=object)}, {'logits': array([-2.9116926], dtype=float32), 'logistic': array([0.05157857], dtype=float32), 'probabili
# ties': array([0.9484214 , 0.05157857], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_
# class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.3637314], dtype=float32), 'logistic': a
# rray([0.0859805], dtype=float32), 'probabilities': array([0.9140195, 0.0859805], dtype=float32), 'class_ids': array([0], dtype=int64), 'cl
# asses': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([
# -2.6850643], dtype=float32), 'logistic': array([0.06386045], dtype=float32), 'probabilities': array([0.9361396 , 0.06386045], dtype=float3
# 2), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'
# 0', b'1'], dtype=object)}, {'logits': array([1.4028249], dtype=float32), 'logistic': array([0.80263174], dtype=float32), 'probabilities':
# array([0.19736823, 0.8026318 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_i
# ds': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.0280151], dtype=float32), 'logistic': array([0
# .04617617], dtype=float32), 'probabilities': array([0.95382386, 0.04617617], dtype=float32), 'class_ids': array([0], dtype=int64), 'classe
# s': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.13
# 744344], dtype=float32), 'logistic': array([0.5343068], dtype=float32), 'probabilities': array([0.46569315, 0.5343069 ], dtype=float32), '
# class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b
# '1'], dtype=object)}, {'logits': array([-2.571899], dtype=float32), 'logistic': array([0.070969], dtype=float32), 'probabilities': array([
# 0.92903095, 0.070969  ], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': ar
# ray([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.107871], dtype=float32), 'logistic': array([0.248268]
# , dtype=float32), 'probabilities': array([0.751732, 0.248268], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0
# '], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.8135146], dtype
# =float32), 'logistic': array([0.14021389], dtype=float32), 'probabilities': array([0.8597861 , 0.14021389], dtype=float32), 'class_ids': a
# rray([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=o
# bject)}, {'logits': array([-2.4909055], dtype=float32), 'logistic': array([0.0764982], dtype=float32), 'probabilities': array([0.9235018,
# 0.0764982], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]),
# 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.0122957], dtype=float32), 'logistic': array([0.7334692], dtype=floa
# t32), 'probabilities': array([0.2665308, 0.7334692], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=
# object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6780133], dtype=float32),
#  'logistic': array([0.06428327], dtype=float32), 'probabilities': array([0.93571675, 0.06428327], dtype=float32), 'class_ids': array([0],
# dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {
# 'logits': array([-1.8272387], dtype=float32), 'logistic': array([0.13856755], dtype=float32), 'probabilities': array([0.8614324 , 0.138567
# 55], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cl
# asses': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.739306], dtype=float32), 'logistic': array([0.32315594], dtype=float32),
# 'probabilities': array([0.67684406, 0.3231559 ], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=obje
# ct), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.3044157], dtype=float32), 'lo
# gistic': array([0.2134228], dtype=float32), 'probabilities': array([0.78657717, 0.2134228 ], dtype=float32), 'class_ids': array([0], dtype
# =int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logi
# ts': array([-2.888566], dtype=float32), 'logistic': array([0.05272169], dtype=float32), 'probabilities': array([0.9472783 , 0.05272169], d
# type=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes'
# : array([b'0', b'1'], dtype=object)}, {'logits': array([-1.7900978], dtype=float32), 'logistic': array([0.14306073], dtype=float32), 'prob
# abilities': array([0.8569393 , 0.14306073], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object),
# 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.5423012], dtype=float32), 'logisti
# c': array([0.0281323], dtype=float32), 'probabilities': array([0.9718677, 0.0281323], dtype=float32), 'class_ids': array([0], dtype=int64)
# , 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': ar
# ray([-3.0344634], dtype=float32), 'logistic': array([0.04589299], dtype=float32), 'probabilities': array([0.954107  , 0.04589299], dtype=f
# loat32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': arra
# y([b'0', b'1'], dtype=object)}, {'logits': array([-2.5610757], dtype=float32), 'logistic': array([0.07168593], dtype=float32), 'probabilit
# ies': array([0.9283141 , 0.07168593], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_c
# lass_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.7961679], dtype=float32), 'logistic': ar
# ray([0.05753161], dtype=float32), 'probabilities': array([0.9424684 , 0.05753161], dtype=float32), 'class_ids': array([0], dtype=int64), '
# classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array
# ([-2.2181253], dtype=float32), 'logistic': array([0.09813459], dtype=float32), 'probabilities': array([0.90186536, 0.09813459], dtype=floa
# t32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([
# b'0', b'1'], dtype=object)}, {'logits': array([2.1137164], dtype=float32), 'logistic': array([0.8922292], dtype=float32), 'probabilities':
#  array([0.10777079, 0.8922292 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_
# ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1013904], dtype=float32), 'logistic': array([
# 0.10896176], dtype=float32), 'probabilities': array([0.89103824, 0.10896176], dtype=float32), 'class_ids': array([0], dtype=int64), 'class
# es': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.
# 9504418], dtype=float32), 'logistic': array([0.04971564], dtype=float32), 'probabilities': array([0.9502844 , 0.04971564], dtype=float32),
#  'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0',
#  b'1'], dtype=object)}, {'logits': array([-1.9220214], dtype=float32), 'logistic': array([0.12763633], dtype=float32), 'probabilities': ar
# ray([0.8723637 , 0.12763633], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids
# ': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.1367986], dtype=float32), 'logistic': array([0.4
# 6585357], dtype=float32), 'probabilities': array([0.5341464 , 0.46585357], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes'
# : array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.988
# 7117], dtype=float32), 'logistic': array([0.12039322], dtype=float32), 'probabilities': array([0.87960684, 0.12039322], dtype=float32), 'c
# lass_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'
# 1'], dtype=object)}, {'logits': array([1.0183829], dtype=float32), 'logistic': array([0.73465747], dtype=float32), 'probabilities': array(
# [0.2653425 , 0.73465747], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': a
# rray([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1832695], dtype=float32), 'logistic': array([0.10126
# 299], dtype=float32), 'probabilities': array([0.898737  , 0.10126299], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': ar
# ray([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.0936798
# 9], dtype=float32), 'logistic': array([0.47659713], dtype=float32), 'probabilities': array([0.52340287, 0.47659716], dtype=float32), 'clas
# s_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1']
# , dtype=object)}, {'logits': array([-1.2909468], dtype=float32), 'logistic': array([0.2156926], dtype=float32), 'probabilities': array([0.
# 7843074 , 0.21569258], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': arra
# y([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.1116831], dtype=float32), 'logistic': array([0.0426279]
# , dtype=float32), 'probabilities': array([0.95737207, 0.0426279 ], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array(
# [b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.9780674], d
# type=float32), 'logistic': array([0.04842661], dtype=float32), 'probabilities': array([0.95157343, 0.04842661], dtype=float32), 'class_ids
# ': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dty
# pe=object)}, {'logits': array([-0.02045248], dtype=float32), 'logistic': array([0.49488705], dtype=float32), 'probabilities': array([0.505
# 11295, 0.49488705], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([
# 0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.5415618], dtype=float32), 'logistic': array([0.82369167], d
# type=float32), 'probabilities': array([0.17630835, 0.82369167], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'
# 1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.5077966], dtyp
# e=float32), 'logistic': array([0.18126556], dtype=float32), 'probabilities': array([0.8187344 , 0.18126556], dtype=float32), 'class_ids':
# array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=
# object)}, {'logits': array([-4.040578], dtype=float32), 'logistic': array([0.01728334], dtype=float32), 'probabilities': array([0.9827167
# , 0.01728334], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]
# ), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.5999875], dtype=float32), 'logistic': array([0.06913922], dtype
# =float32), 'probabilities': array([0.93086076, 0.06913922], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'],
#  dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.5457358], dtype=fl
# oat32), 'logistic': array([0.07271348], dtype=float32), 'probabilities': array([0.92728657, 0.07271348], dtype=float32), 'class_ids': arra
# y([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=obje
# ct)}, {'logits': array([-3.0256624], dtype=float32), 'logistic': array([0.0462799], dtype=float32), 'probabilities': array([0.9537201, 0.0
# 462799], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'al
# l_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.7169724], dtype=float32), 'logistic': array([0.06197926], dtype=float
# 32), 'probabilities': array([0.93802077, 0.06197926], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype
# =object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6850643], dtype=float32)
# , 'logistic': array([0.06386045], dtype=float32), 'probabilities': array([0.9361396 , 0.06386045], dtype=float32), 'class_ids': array([0],
#  dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)},
# {'logits': array([0.3728447], dtype=float32), 'logistic': array([0.59214616], dtype=float32), 'probabilities': array([0.4078538 , 0.592146
# 16], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cl
# asses': array([b'0', b'1'], dtype=object)}, {'logits': array([2.5942688], dtype=float32), 'logistic': array([0.9304918], dtype=float32), '
# probabilities': array([0.06950818, 0.9304918 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=objec
# t), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.05302428], dtype=float32), 'lo
# gistic': array([0.48674703], dtype=float32), 'probabilities': array([0.513253  , 0.48674706], dtype=float32), 'class_ids': array([0], dtyp
# e=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'log
# its': array([-1.2125809], dtype=float32), 'logistic': array([0.22924471], dtype=float32), 'probabilities': array([0.7707553, 0.2292447], d
# type=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes'
# : array([b'0', b'1'], dtype=object)}, {'logits': array([-2.178401], dtype=float32), 'logistic': array([0.10170691], dtype=float32), 'proba
# bilities': array([0.898293  , 0.10170691], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), '
# all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.1329489], dtype=float32), 'logistic'
# : array([0.8940646], dtype=float32), 'probabilities': array([0.10593537, 0.89406466], dtype=float32), 'class_ids': array([1], dtype=int64)
# , 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': ar
# ray([-2.882414], dtype=float32), 'logistic': array([0.05302978], dtype=float32), 'probabilities': array([0.9469702 , 0.05302978], dtype=fl
# oat32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array
# ([b'0', b'1'], dtype=object)}, {'logits': array([-0.56525], dtype=float32), 'logistic': array([0.3623336], dtype=float32), 'probabilities'
# : array([0.6376664, 0.3623336], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_i
# ds': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.13246296], dtype=float32), 'logistic': array([
# 0.46693262], dtype=float32), 'probabilities': array([0.5330674, 0.4669326], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes
# ': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.15
# 03816], dtype=float32), 'logistic': array([0.04107624], dtype=float32), 'probabilities': array([0.9589237 , 0.04107624], dtype=float32), '
# class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b
# '1'], dtype=object)}, {'logits': array([0.87087375], dtype=float32), 'logistic': array([0.70492744], dtype=float32), 'probabilities': arra
# y([0.29507253, 0.70492744], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids':
#  array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.95154995], dtype=float32), 'logistic': array([0.721
# 4268], dtype=float32), 'probabilities': array([0.27857324, 0.7214268 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': a
# rray([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.5901527
# ], dtype=float32), 'logistic': array([0.93022513], dtype=float32), 'probabilities': array([0.06977487, 0.9302251 ], dtype=float32), 'class
# _ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'],
#  dtype=object)}, {'logits': array([-2.3328624], dtype=float32), 'logistic': array([0.08843763], dtype=float32), 'probabilities': array([0.
# 9115624 , 0.08843764], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': arra
# y([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.7738223], dtype=float32), 'logistic': array([0.05875527
# ], dtype=float32), 'probabilities': array([0.94124466, 0.05875527], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array
# ([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.1808114], d
# type=float32), 'logistic': array([0.8985131], dtype=float32), 'probabilities': array([0.10148691, 0.8985131 ], dtype=float32), 'class_ids'
# : array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtyp
# e=object)}, {'logits': array([-0.47336602], dtype=float32), 'logistic': array([0.38381985], dtype=float32), 'probabilities': array([0.6161
# 801 , 0.38381985], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0
# , 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.5208113], dtype=float32), 'logistic': array([0.62733746], dt
# ype=float32), 'probabilities': array([0.3726625 , 0.62733746], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1
# '], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.7980176], dtype
# =float32), 'logistic': array([0.14209256], dtype=float32), 'probabilities': array([0.8579075 , 0.14209256], dtype=float32), 'class_ids': a
# rray([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=o
# bject)}, {'logits': array([-3.4939609], dtype=float32), 'logistic': array([0.02948455], dtype=float32), 'probabilities': array([0.9705155
# , 0.02948455], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]
# ), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.7537277], dtype=float32), 'logistic': array([0.05987648], dtype
# =float32), 'probabilities': array([0.94012356, 0.05987648], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'],
#  dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.6413642], dtype=fl
# oat32), 'logistic': array([0.16227952], dtype=float32), 'probabilities': array([0.8377205 , 0.16227952], dtype=float32), 'class_ids': arra
# y([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=obje
# ct)}, {'logits': array([-3.029313], dtype=float32), 'logistic': array([0.04611904], dtype=float32), 'probabilities': array([0.953881  , 0.
# 04611904], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), '
# all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.04585634], dtype=float32), 'logistic': array([0.4885379], dtype=flo
# at32), 'probabilities': array([0.5114621 , 0.48853794], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dty
# pe=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.80403996], dtype=float
# 32), 'logistic': array([0.309162], dtype=float32), 'probabilities': array([0.690838, 0.309162], dtype=float32), 'class_ids': array([0], dt
# ype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'l
# ogits': array([0.01370861], dtype=float32), 'logistic': array([0.5034271], dtype=float32), 'probabilities': array([0.49657288, 0.5034271 ]
# , dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_class
# es': array([b'0', b'1'], dtype=object)}, {'logits': array([0.02220322], dtype=float32), 'logistic': array([0.5055506], dtype=float32), 'pr
# obabilities': array([0.4944494 , 0.50555056], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object)
# , 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.0474293], dtype=float32), 'logis
# tic': array([0.11431239], dtype=float32), 'probabilities': array([0.88568765, 0.11431239], dtype=float32), 'class_ids': array([0], dtype=i
# nt64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits
# ': array([1.0698769], dtype=float32), 'logistic': array([0.74457353], dtype=float32), 'probabilities': array([0.2554265 , 0.74457353], dty
# pe=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes':
# array([b'0', b'1'], dtype=object)}, {'logits': array([0.71586025], dtype=float32), 'logistic': array([0.67169476], dtype=float32), 'probab
# ilities': array([0.32830524, 0.67169476], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'a
# ll_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.28862858], dtype=float32), 'logistic'
# : array([0.57166034], dtype=float32), 'probabilities': array([0.42833963, 0.57166034], dtype=float32), 'class_ids': array([1], dtype=int64
# ), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': a
# rray([-3.2571068], dtype=float32), 'logistic': array([0.03707235], dtype=float32), 'probabilities': array([0.9629277 , 0.03707235], dtype=
# float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': arr
# ay([b'0', b'1'], dtype=object)}, {'logits': array([0.59935355], dtype=float32), 'logistic': array([0.64550835], dtype=float32), 'probabili
# ties': array([0.3544916, 0.6455084], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_cl
# ass_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.08452214], dtype=float32), 'logistic': arr
# ay([0.521118], dtype=float32), 'probabilities': array([0.47888207, 0.521118  ], dtype=float32), 'class_ids': array([1], dtype=int64), 'cla
# sses': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0
# .10231794], dtype=float32), 'logistic': array([0.5255572], dtype=float32), 'probabilities': array([0.47444278, 0.52555716], dtype=float32)
# , 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0'
# , b'1'], dtype=object)}, {'logits': array([-3.8456454], dtype=float32), 'logistic': array([0.02092537], dtype=float32), 'probabilities': a
# rray([0.97907466, 0.02092537], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_id
# s': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.3625903], dtype=float32), 'logistic': array([0.7
# 9618037], dtype=float32), 'probabilities': array([0.20381962, 0.79618037], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes'
# : array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.1352
# 9374], dtype=float32), 'logistic': array([0.53377193], dtype=float32), 'probabilities': array([0.46622807, 0.53377193], dtype=float32), 'c
# lass_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'
# 1'], dtype=object)}, {'logits': array([-2.9893875], dtype=float32), 'logistic': array([0.04790762], dtype=float32), 'probabilities': array
# ([0.9520924 , 0.04790762], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids':
# array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.0330181], dtype=float32), 'logistic': array([0.4917
# 4622], dtype=float32), 'probabilities': array([0.5082538 , 0.49174625], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': a
# rray([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.95149]
# , dtype=float32), 'logistic': array([0.04966614], dtype=float32), 'probabilities': array([0.95033383, 0.04966614], dtype=float32), 'class_
# ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'],
# dtype=object)}, {'logits': array([0.6402417], dtype=float32), 'logistic': array([0.6548081], dtype=float32), 'probabilities': array([0.345
# 19193, 0.6548081 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([
# 0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.3875177], dtype=float32), 'logistic': array([0.40431502],
# dtype=float32), 'probabilities': array([0.595685, 0.404315], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0']
# , dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.8146732], dtype=f
# loat32), 'logistic': array([0.0565364], dtype=float32), 'probabilities': array([0.9434636, 0.0565364], dtype=float32), 'class_ids': array(
# [0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object
# )}, {'logits': array([-0.70011854], dtype=float32), 'logistic': array([0.33178595], dtype=float32), 'probabilities': array([0.668214  , 0.
# 33178595], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), '
# all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.1462194], dtype=float32), 'logistic': array([0.7588197], dtype=float
# 32), 'probabilities': array([0.2411803 , 0.75881964], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype
# =object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.911629], dtype=float32),
#  'logistic': array([0.05158168], dtype=float32), 'probabilities': array([0.94841826, 0.05158168], dtype=float32), 'class_ids': array([0],
# dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {
# 'logits': array([-2.950192], dtype=float32), 'logistic': array([0.04972744], dtype=float32), 'probabilities': array([0.95027256, 0.0497274
# 4], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cla
# sses': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.26018977], dtype=float32), 'logistic': array([0.43531704], dtype=float32),
#  'probabilities': array([0.56468296, 0.43531707], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=obj
# ect), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6008482], dtype=float32), 'l
# ogistic': array([0.06908385], dtype=float32), 'probabilities': array([0.93091613, 0.06908385], dtype=float32), 'class_ids': array([0], dty
# pe=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'lo
# gits': array([-2.9185338], dtype=float32), 'logistic': array([0.05124494], dtype=float32), 'probabilities': array([0.9487551 , 0.05124494]
# , dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_class
# es': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1613925], dtype=float32), 'logistic': array([0.10327143], dtype=float32), 'p
# robabilities': array([0.8967286 , 0.10327143], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object
# ), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.3383145], dtype=float32), 'logi
# stic': array([0.03427991], dtype=float32), 'probabilities': array([0.9657201 , 0.03427991], dtype=float32), 'class_ids': array([0], dtype=
# int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logit
# s': array([0.02640106], dtype=float32), 'logistic': array([0.5065999], dtype=float32), 'probabilities': array([0.49340016, 0.5065999 ], dt
# ype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes':
#  array([b'0', b'1'], dtype=object)}, {'logits': array([-2.7175808], dtype=float32), 'logistic': array([0.06194389], dtype=float32), 'proba
# bilities': array([0.9380561 , 0.06194389], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), '
# all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.8062472], dtype=float32), 'logistic
# ': array([0.02174796], dtype=float32), 'probabilities': array([0.97825205, 0.02174796], dtype=float32), 'class_ids': array([0], dtype=int6
# 4), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits':
# array([-3.0280151], dtype=float32), 'logistic': array([0.04617617], dtype=float32), 'probabilities': array([0.95382386, 0.04617617], dtype
# =float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': ar
# ray([b'0', b'1'], dtype=object)}, {'logits': array([-2.6791086], dtype=float32), 'logistic': array([0.06421743], dtype=float32), 'probabil
# ities': array([0.9357826 , 0.06421743], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all
# _class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6992779], dtype=float32), 'logistic':
# array([0.06301598], dtype=float32), 'probabilities': array([0.936984  , 0.06301598], dtype=float32), 'class_ids': array([0], dtype=int64),
#  'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': arr
# ay([-2.3417754], dtype=float32), 'logistic': array([0.08772174], dtype=float32), 'probabilities': array([0.9122783 , 0.08772174], dtype=fl
# oat32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array
# ([b'0', b'1'], dtype=object)}, {'logits': array([1.2800951], dtype=float32), 'logistic': array([0.78246593], dtype=float32), 'probabilitie
# s': array([0.21753405, 0.782466  ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_cla
# ss_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.938754], dtype=float32), 'logistic': array
# ([0.05027073], dtype=float32), 'probabilities': array([0.94972926, 0.05027073], dtype=float32), 'class_ids': array([0], dtype=int64), 'cla
# sses': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-
# 2.7949579], dtype=float32), 'logistic': array([0.05759725], dtype=float32), 'probabilities': array([0.9424028 , 0.05759725], dtype=float32
# ), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0
# ', b'1'], dtype=object)}, {'logits': array([0.139179], dtype=float32), 'logistic': array([0.5347387], dtype=float32), 'probabilities': arr
# ay([0.4652613 , 0.53473866], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids'
# : array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.0863245], dtype=float32), 'logistic': array([0.889
# 56684], dtype=float32), 'probabilities': array([0.11043313, 0.88956684], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes':
# array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.92670
# 82], dtype=float32), 'logistic': array([0.05084896], dtype=float32), 'probabilities': array([0.949151  , 0.05084896], dtype=float32), 'cla
# ss_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'
# ], dtype=object)}, {'logits': array([-2.63976], dtype=float32), 'logistic': array([0.06662296], dtype=float32), 'probabilities': array([0.
# 9333771 , 0.06662296], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': arra
# y([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.950192], dtype=float32), 'logistic': array([0.04972744]
# , dtype=float32), 'probabilities': array([0.95027256, 0.04972744], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array(
# [b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.1384323], d
# type=float32), 'logistic': array([0.24260831], dtype=float32), 'probabilities': array([0.7573917 , 0.24260832], dtype=float32), 'class_ids
# ': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dty
# pe=object)}, {'logits': array([-2.5742416], dtype=float32), 'logistic': array([0.07081469], dtype=float32), 'probabilities': array([0.9291
# 8533, 0.0708147 ], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0
# , 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.3535101], dtype=float32), 'logistic': array([0.08678717], d
# type=float32), 'probabilities': array([0.91321284, 0.08678717], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'
# 0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.3784223], dtyp
# e=float32), 'logistic': array([0.03297667], dtype=float32), 'probabilities': array([0.9670233 , 0.03297667], dtype=float32), 'class_ids':
# array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=
# object)}, {'logits': array([-2.3535032], dtype=float32), 'logistic': array([0.08678772], dtype=float32), 'probabilities': array([0.9132122
# 4, 0.08678772], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1
# ]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.9397278], dtype=float32), 'logistic': array([0.05022426], dtyp
# e=float32), 'probabilities': array([0.9497757 , 0.05022426], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0']
# , dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.255628], dtype=fl
# oat32), 'logistic': array([0.09486509], dtype=float32), 'probabilities': array([0.90513486, 0.09486509], dtype=float32), 'class_ids': arra
# y([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=obje
# ct)}, {'logits': array([-3.2142105], dtype=float32), 'logistic': array([0.03863445], dtype=float32), 'probabilities': array([0.9613656 , 0
# .03863445], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]),
# 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.1568905], dtype=float32), 'logistic': array([0.23923276], dtype=fl
# oat32), 'probabilities': array([0.7607673 , 0.23923276], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dt
# ype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1200404], dtype=float
# 32), 'logistic': array([0.1071642], dtype=float32), 'probabilities': array([0.89283574, 0.1071642 ], dtype=float32), 'class_ids': array([0
# ], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}
# , {'logits': array([-3.1354952], dtype=float32), 'logistic': array([0.04166663], dtype=float32), 'probabilities': array([0.9583333 , 0.041
# 66663], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all
# _classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.1394906], dtype=float32), 'logistic': array([0.10531738], dtype=float3
# 2), 'probabilities': array([0.89468265, 0.10531738], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=
# object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.20996855], dtype=float32)
# , 'logistic': array([0.44769984], dtype=float32), 'probabilities': array([0.55230016, 0.44769987], dtype=float32), 'class_ids': array([0],
#  dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)},
# {'logits': array([-2.3535032], dtype=float32), 'logistic': array([0.08678772], dtype=float32), 'probabilities': array([0.91321224, 0.08678
# 772], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_c
# lasses': array([b'0', b'1'], dtype=object)}, {'logits': array([0.13937499], dtype=float32), 'logistic': array([0.5347874], dtype=float32),
#  'probabilities': array([0.46521258, 0.5347875 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=obj
# ect), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.5895233], dtype=float32), 'l
# ogistic': array([0.06981573], dtype=float32), 'probabilities': array([0.93018425, 0.06981573], dtype=float32), 'class_ids': array([0], dty
# pe=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'lo
# gits': array([1.552328], dtype=float32), 'logistic': array([0.82524973], dtype=float32), 'probabilities': array([0.1747503 , 0.82524973],
# dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes
# ': array([b'0', b'1'], dtype=object)}, {'logits': array([0.41767776], dtype=float32), 'logistic': array([0.60292745], dtype=float32), 'pro
# babilities': array([0.39707258, 0.60292745], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object),
#  'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.106041], dtype=float32), 'logisti
# c': array([0.04285876], dtype=float32), 'probabilities': array([0.9571413 , 0.04285876], dtype=float32), 'class_ids': array([0], dtype=int
# 64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits':
#  array([1.5586094], dtype=float32), 'logistic': array([0.8261537], dtype=float32), 'probabilities': array([0.17384629, 0.82615376], dtype=
# float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': arr
# ay([b'0', b'1'], dtype=object)}, {'logits': array([-3.1309133], dtype=float32), 'logistic': array([0.04184997], dtype=float32), 'probabili
# ties': array([0.95814997, 0.04184997], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_
# class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.351097], dtype=float32), 'logistic': ar
# ray([0.08697861], dtype=float32), 'probabilities': array([0.91302145, 0.08697861], dtype=float32), 'class_ids': array([0], dtype=int64), '
# classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array
# ([2.0075061], dtype=float32), 'logistic': array([0.8815829], dtype=float32), 'probabilities': array([0.11841707, 0.8815829 ], dtype=float3
# 2), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'
# 0', b'1'], dtype=object)}, {'logits': array([-2.7167222], dtype=float32), 'logistic': array([0.0619938], dtype=float32), 'probabilities':
# array([0.93800616, 0.06199379], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_i
# ds': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.6189872], dtype=float32), 'logistic': array([0.
# 6499882], dtype=float32), 'probabilities': array([0.35001183, 0.6499882 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes'
# : array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.950
# 4418], dtype=float32), 'logistic': array([0.04971564], dtype=float32), 'probabilities': array([0.9502844 , 0.04971564], dtype=float32), 'c
# lass_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'
# 1'], dtype=object)}, {'logits': array([-1.7867864], dtype=float32), 'logistic': array([0.14346716], dtype=float32), 'probabilities': array
# ([0.8565328 , 0.14346717], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids':
# array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.5727777], dtype=float32), 'logistic': array([0.0273
# 1092], dtype=float32), 'probabilities': array([0.9726891 , 0.02731092], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': a
# rray([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.600848
# 2], dtype=float32), 'logistic': array([0.06908385], dtype=float32), 'probabilities': array([0.93091613, 0.06908385], dtype=float32), 'clas
# s_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1']
# , dtype=object)}, {'logits': array([-0.09985577], dtype=float32), 'logistic': array([0.47505677], dtype=float32), 'probabilities': array([
# 0.52494323, 0.47505677], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': ar
# ray([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.8841826], dtype=float32), 'logistic': array([0.8680908
# ], dtype=float32), 'probabilities': array([0.13190919, 0.8680908 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array
# ([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.2836406],
# dtype=float32), 'logistic': array([0.09248693], dtype=float32), 'probabilities': array([0.9075131 , 0.09248693], dtype=float32), 'class_id
# s': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dt
# ype=object)}, {'logits': array([-1.6329238], dtype=float32), 'logistic': array([0.16343021], dtype=float32), 'probabilities': array([0.836
# 5698 , 0.16343021], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([
# 0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6781282], dtype=float32), 'logistic': array([0.06427637],
# dtype=float32), 'probabilities': array([0.93572366, 0.06427637], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b
# '0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6647506], dty
# pe=float32), 'logistic': array([0.06508566], dtype=float32), 'probabilities': array([0.93491435, 0.06508566], dtype=float32), 'class_ids':
#  array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype
# =object)}, {'logits': array([-1.8555738], dtype=float32), 'logistic': array([0.1352198], dtype=float32), 'probabilities': array([0.8647802
# , 0.1352198], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1])
# , 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.068892], dtype=float32), 'logistic': array([0.11215732], dtype=f
# loat32), 'probabilities': array([0.8878427 , 0.11215732], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], d
# type=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.3451474], dtype=float
# 32), 'logistic': array([0.91254777], dtype=float32), 'probabilities': array([0.08745226, 0.91254777], dtype=float32), 'class_ids': array([
# 1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)
# }, {'logits': array([-0.2696526], dtype=float32), 'logistic': array([0.4329924], dtype=float32), 'probabilities': array([0.5670076 , 0.432
# 99237], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all
# _classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.6002374], dtype=float32), 'logistic': array([0.06912315], dtype=float3
# 2), 'probabilities': array([0.93087685, 0.06912315], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=
# object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.8194447], dtype=float32),
#  'logistic': array([0.13950051], dtype=float32), 'probabilities': array([0.86049944, 0.13950051], dtype=float32), 'class_ids': array([0],
# dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {
# 'logits': array([0.1395035], dtype=float32), 'logistic': array([0.5348194], dtype=float32), 'probabilities': array([0.4651806, 0.5348194],
#  dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classe
# s': array([b'0', b'1'], dtype=object)}, {'logits': array([0.5268687], dtype=float32), 'logistic': array([0.6287525], dtype=float32), 'prob
# abilities': array([0.3712475 , 0.62875247], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object),
# 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.9457442], dtype=float32), 'logisti
# c': array([0.12501815], dtype=float32), 'probabilities': array([0.8749818 , 0.12501815], dtype=float32), 'class_ids': array([0], dtype=int
# 64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits':
#  array([-1.9594754], dtype=float32), 'logistic': array([0.12352384], dtype=float32), 'probabilities': array([0.87647617, 0.12352383], dtyp
# e=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': a
# rray([b'0', b'1'], dtype=object)}, {'logits': array([-2.8204494], dtype=float32), 'logistic': array([0.05622908], dtype=float32), 'probabi
# lities': array([0.94377095, 0.05622908], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'al
# l_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.5045087], dtype=float32), 'logistic':
#  array([0.07554271], dtype=float32), 'probabilities': array([0.9244573, 0.0755427], dtype=float32), 'class_ids': array([0], dtype=int64),
# 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': arra
# y([1.145669], dtype=float32), 'logistic': array([0.7587189], dtype=float32), 'probabilities': array([0.24128106, 0.75871897], dtype=float3
# 2), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'
# 0', b'1'], dtype=object)}, {'logits': array([-3.7786455], dtype=float32), 'logistic': array([0.02234301], dtype=float32), 'probabilities':
#  array([0.977657  , 0.02234301], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_
# ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.814001], dtype=float32), 'logistic': array([0
# .05657226], dtype=float32), 'probabilities': array([0.9434278 , 0.05657226], dtype=float32), 'class_ids': array([0], dtype=int64), 'classe
# s': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.7
# 7945566], dtype=float32), 'logistic': array([0.3144372], dtype=float32), 'probabilities': array([0.6855628, 0.3144372], dtype=float32), 'c
# lass_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'
# 1'], dtype=object)}, {'logits': array([0.1400037], dtype=float32), 'logistic': array([0.5349439], dtype=float32), 'probabilities': array([
# 0.46505615, 0.5349439 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': ar
# ray([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.5458783], dtype=float32), 'logistic': array([0.175682
# 37], dtype=float32), 'probabilities': array([0.82431763, 0.17568237], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': arr
# ay([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.8804154]
# , dtype=float32), 'logistic': array([0.05313023], dtype=float32), 'probabilities': array([0.94686973, 0.05313023], dtype=float32), 'class_
# ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'],
# dtype=object)}, {'logits': array([-2.2367754], dtype=float32), 'logistic': array([0.09649631], dtype=float32), 'probabilities': array([0.9
# 035037 , 0.09649631], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array
# ([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.0733695], dtype=float32), 'logistic': array([0.0442192],
#  dtype=float32), 'probabilities': array([0.9557808, 0.0442192], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'
# 0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.5727777], dtyp
# e=float32), 'logistic': array([0.02731092], dtype=float32), 'probabilities': array([0.9726891 , 0.02731092], dtype=float32), 'class_ids':
# array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=
# object)}, {'logits': array([-3.0281081], dtype=float32), 'logistic': array([0.04617208], dtype=float32), 'probabilities': array([0.953828
#  , 0.04617208], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1
# ]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([2.4175675], dtype=float32), 'logistic': array([0.91815716], dtype
# =float32), 'probabilities': array([0.08184285, 0.91815716], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'],
#  dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.73963606], dtype=fl
# oat32), 'logistic': array([0.67691624], dtype=float32), 'probabilities': array([0.32308376, 0.6769163 ], dtype=float32), 'class_ids': arra
# y([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=obje
# ct)}, {'logits': array([-2.8534565], dtype=float32), 'logistic': array([0.05450292], dtype=float32), 'probabilities': array([0.9454971 , 0
# .05450292], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]),
# 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.06076466], dtype=float32), 'logistic': array([0.4848135], dtype=fl
# oat32), 'probabilities': array([0.5151865, 0.4848135], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtyp
# e=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.3719019], dtype=float32
# ), 'logistic': array([0.20231274], dtype=float32), 'probabilities': array([0.79768723, 0.20231274], dtype=float32), 'class_ids': array([0]
# , dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)},
#  {'logits': array([-3.0741806], dtype=float32), 'logistic': array([0.04418493], dtype=float32), 'probabilities': array([0.9558151 , 0.0441
# 8493], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_
# classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.5045497], dtype=float32), 'logistic': array([0.07553984], dtype=float32
# ), 'probabilities': array([0.9244602 , 0.07553984], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=o
# bject), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.3737717], dtype=float32),
# 'logistic': array([0.08519473], dtype=float32), 'probabilities': array([0.9148053 , 0.08519474], dtype=float32), 'class_ids': array([0], d
# type=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'
# logits': array([-2.6379054], dtype=float32), 'logistic': array([0.06673838], dtype=float32), 'probabilities': array([0.93326163, 0.0667383
# 8], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cla
# sses': array([b'0', b'1'], dtype=object)}, {'logits': array([0.13935472], dtype=float32), 'logistic': array([0.53478235], dtype=float32),
# 'probabilities': array([0.4652176, 0.5347824], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object
# ), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.9565878], dtype=float32), 'logi
# stic': array([0.04942608], dtype=float32), 'probabilities': array([0.950574  , 0.04942608], dtype=float32), 'class_ids': array([0], dtype=
# int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logit
# s': array([-0.90066504], dtype=float32), 'logistic': array([0.28891385], dtype=float32), 'probabilities': array([0.71108615, 0.28891385],
# dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes
# ': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.1439819], dtype=float32), 'logistic': array([0.24159002], dtype=float32), 'pro
# babilities': array([0.75841   , 0.24159002], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object),
#  'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.0671294], dtype=float32), 'logist
# ic': array([0.04448368], dtype=float32), 'probabilities': array([0.95551634, 0.04448368], dtype=float32), 'class_ids': array([0], dtype=in
# t64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits'
# : array([-2.296811], dtype=float32), 'logistic': array([0.09138741], dtype=float32), 'probabilities': array([0.9086126 , 0.09138741], dtyp
# e=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': a
# rray([b'0', b'1'], dtype=object)}, {'logits': array([0.52209806], dtype=float32), 'logistic': array([0.6276383], dtype=float32), 'probabil
# ities': array([0.37236178, 0.6276382 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all
# _class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.7370714], dtype=float32), 'logistic':
# array([0.32364488], dtype=float32), 'probabilities': array([0.6763551 , 0.32364488], dtype=float32), 'class_ids': array([0], dtype=int64),
#  'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': arr
# ay([-2.7175832], dtype=float32), 'logistic': array([0.06194375], dtype=float32), 'probabilities': array([0.93805623, 0.06194375], dtype=fl
# oat32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array
# ([b'0', b'1'], dtype=object)}, {'logits': array([-0.9077657], dtype=float32), 'logistic': array([0.28745726], dtype=float32), 'probabiliti
# es': array([0.7125427 , 0.28745726], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_cl
# ass_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.7536198], dtype=float32), 'logistic': arr
# ay([0.14759122], dtype=float32), 'probabilities': array([0.8524088 , 0.14759122], dtype=float32), 'class_ids': array([0], dtype=int64), 'c
# lasses': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array(
# [1.4248272], dtype=float32), 'logistic': array([0.80609405], dtype=float32), 'probabilities': array([0.19390595, 0.80609405], dtype=float3
# 2), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'
# 0', b'1'], dtype=object)}, {'logits': array([1.1957444], dtype=float32), 'logistic': array([0.7677669], dtype=float32), 'probabilities': a
# rray([0.23223314, 0.7677669 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_id
# s': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.4836988], dtype=float32), 'logistic': array([0.
# 07700888], dtype=float32), 'probabilities': array([0.92299104, 0.07700887], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes
# ': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.042
# 09591], dtype=float32), 'logistic': array([0.5105224], dtype=float32), 'probabilities': array([0.4894776, 0.5105224], dtype=float32), 'cla
# ss_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'
# ], dtype=object)}, {'logits': array([-2.792193], dtype=float32), 'logistic': array([0.05774752], dtype=float32), 'probabilities': array([0
# .94225246, 0.05774751], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': arr
# ay([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.3535101], dtype=float32), 'logistic': array([0.0867871
# 7], dtype=float32), 'probabilities': array([0.91321284, 0.08678717], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': arra
# y([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.35072958]
# , dtype=float32), 'logistic': array([0.4132055], dtype=float32), 'probabilities': array([0.5867945 , 0.41320553], dtype=float32), 'class_i
# ds': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], d
# type=object)}, {'logits': array([-2.6008687], dtype=float32), 'logistic': array([0.06908254], dtype=float32), 'probabilities': array([0.93
# 091744, 0.06908254], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array(
# [0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.1903195], dtype=float32), 'logistic': array([0.7667982], d
# type=float32), 'probabilities': array([0.23320177, 0.7667982 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'
# 1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.97039384], dtyp
# e=float32), 'logistic': array([0.72519803], dtype=float32), 'probabilities': array([0.274802, 0.725198], dtype=float32), 'class_ids': arra
# y([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=obje
# ct)}, {'logits': array([-2.834755], dtype=float32), 'logistic': array([0.05547472], dtype=float32), 'probabilities': array([0.9445253 , 0.
# 05547472], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), '
# all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.17824842], dtype=float32), 'logistic': array([0.54444444], dtype=flo
# at32), 'probabilities': array([0.45555553, 0.5444445 ], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dty
# pe=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.314252], dtype=float32
# ), 'logistic': array([0.08994948], dtype=float32), 'probabilities': array([0.9100505 , 0.08994949], dtype=float32), 'class_ids': array([0]
# , dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)},
#  {'logits': array([-1.3261539], dtype=float32), 'logistic': array([0.20979626], dtype=float32), 'probabilities': array([0.7902037 , 0.2097
# 9626], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_
# classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.9130244], dtype=float32), 'logistic': array([0.05151346], dtype=float32
# ), 'probabilities': array([0.9484865 , 0.05151346], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=o
# bject), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.17870142], dtype=float32),
# 'logistic': array([0.54455686], dtype=float32), 'probabilities': array([0.45544314, 0.5445568 ], dtype=float32), 'class_ids': array([1], d
# type=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'
# logits': array([-0.17589258], dtype=float32), 'logistic': array([0.45613986], dtype=float32), 'probabilities': array([0.54386014, 0.456139
# 86], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_cl
# asses': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.5364618], dtype=float32), 'logistic': array([0.0282924], dtype=float32),
# 'probabilities': array([0.9717076, 0.0282924], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object
# ), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.2446895], dtype=float32), 'logi
# stic': array([0.09580853], dtype=float32), 'probabilities': array([0.90419143, 0.09580853], dtype=float32), 'class_ids': array([0], dtype=
# int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logit
# s': array([0.77398753], dtype=float32), 'logistic': array([0.68438286], dtype=float32), 'probabilities': array([0.31561717, 0.68438286], d
# type=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes'
# : array([b'0', b'1'], dtype=object)}, {'logits': array([-0.83399206], dtype=float32), 'logistic': array([0.30280164], dtype=float32), 'pro
# babilities': array([0.6971984 , 0.30280164], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object),
#  'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.5646033], dtype=float32), 'logisti
# c': array([0.637517], dtype=float32), 'probabilities': array([0.362483, 0.637517], dtype=float32), 'class_ids': array([1], dtype=int64), '
# classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array
# ([-0.90709656], dtype=float32), 'logistic': array([0.28759435], dtype=float32), 'probabilities': array([0.7124057 , 0.28759435], dtype=flo
# at32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array(
# [b'0', b'1'], dtype=object)}, {'logits': array([-3.0272422], dtype=float32), 'logistic': array([0.04621023], dtype=float32), 'probabilitie
# s': array([0.9537898 , 0.04621023], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_cla
# ss_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.639102], dtype=float32), 'logistic': array
# ([0.06666389], dtype=float32), 'probabilities': array([0.9333361 , 0.06666389], dtype=float32), 'class_ids': array([0], dtype=int64), 'cla
# sses': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-
# 2.5211706], dtype=float32), 'logistic': array([0.0743873], dtype=float32), 'probabilities': array([0.92561275, 0.0743873 ], dtype=float32)
# , 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0'
# , b'1'], dtype=object)}, {'logits': array([-4.740572], dtype=float32), 'logistic': array([0.00865803], dtype=float32), 'probabilities': ar
# ray([0.991342  , 0.00865803], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids
# ': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([1.3056948], dtype=float32), 'logistic': array([0.78
# 679186], dtype=float32), 'probabilities': array([0.21320814, 0.78679186], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes':
#  array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.01765
# 598], dtype=float32), 'logistic': array([0.50441384], dtype=float32), 'probabilities': array([0.4955861 , 0.50441384], dtype=float32), 'cl
# ass_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1
# '], dtype=object)}, {'logits': array([-2.3535032], dtype=float32), 'logistic': array([0.08678772], dtype=float32), 'probabilities': array(
# [0.91321224, 0.08678772], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': a
# rray([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-1.8511245], dtype=float32), 'logistic': array([0.13574
# 092], dtype=float32), 'probabilities': array([0.8642591 , 0.13574092], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': ar
# ray([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.8114051]
# , dtype=float32), 'logistic': array([0.69240886], dtype=float32), 'probabilities': array([0.30759114, 0.69240886], dtype=float32), 'class_
# ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'],
# dtype=object)}, {'logits': array([-1.065562], dtype=float32), 'logistic': array([0.25624797], dtype=float32), 'probabilities': array([0.74
# 3752  , 0.25624797], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array(
# [0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-2.9478393], dtype=float32), 'logistic': array([0.04983873],
#  dtype=float32), 'probabilities': array([0.9501613 , 0.04983874], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([
# b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([0.63756275], dt
# ype=float32), 'logistic': array([0.6542023], dtype=float32), 'probabilities': array([0.3457977 , 0.65420234], dtype=float32), 'class_ids':
#  array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype
# =object)}, {'logits': array([0.6898254], dtype=float32), 'logistic': array([0.66592807], dtype=float32), 'probabilities': array([0.3340718
# 7, 0.66592807], dtype=float32), 'class_ids': array([1], dtype=int64), 'classes': array([b'1'], dtype=object), 'all_class_ids': array([0, 1
# ]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-3.145], dtype=float32), 'logistic': array([0.04128874], dtype=fl
# oat32), 'probabilities': array([0.9587112 , 0.04128874], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dt
# ype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.6878426], dtype=float
# 32), 'logistic': array([0.33451316], dtype=float32), 'probabilities': array([0.6654868 , 0.33451316], dtype=float32), 'class_ids': array([
# 0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)
# }, {'logits': array([-2.1013904], dtype=float32), 'logistic': array([0.10896176], dtype=float32), 'probabilities': array([0.89103824, 0.10
# 896176], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtype=object), 'all_class_ids': array([0, 1]), 'al
# l_classes': array([b'0', b'1'], dtype=object)}, {'logits': array([-0.97363275], dtype=float32), 'logistic': array([0.27415702], dtype=floa
# t32), 'probabilities': array([0.725843  , 0.27415702], dtype=float32), 'class_ids': array([0], dtype=int64), 'classes': array([b'0'], dtyp
# e=object), 'all_class_ids': array([0, 1]), 'all_classes': array([b'0', b'1'], dtype=object)}]

