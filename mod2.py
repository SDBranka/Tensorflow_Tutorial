import tensorflow as tf

# show version of tensorflow used
# print(tf.version)
# <module 'tensorflow._api.v2.version' from 'C:\\Users\\sdbra\\Desktop\\Programmin
# g\\Projects\\Machine_Learning\\TFTutorial\\lib\\site-packages\\tensorflow\\_api\
# \v2\\version\\__init__.py'>


# creating tensors
# var_name = tf.Variable(var, var_type)
string = tf.Variable("This is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)

# rank/degree of tensors
# # the number of dimensions involved in the tensor
# ## rank 0 known as scalar
rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

# to determine the rank of a tensor
tf.rank(rank2_tensor)
# <tf.Tensor: shape=(), dtype=int32, numpy=2>   #numpy value equals the rank

# shape of tensors
rank2_tensor.shape
# <TensorShape([2, 2])>   # 2 rows, 2 columns

# changing shape
tensor1 = tf.ones([1,2,3])   #creates a shape [1,2,3] tensor full of ones
# print(tensor1)
# tf.Tensor(
# [[[1. 1. 1.]
#   [1. 1. 1.]]], shape=(1, 2, 3), dtype=float32)
tensor2 = tf.reshape(tensor1, [2,3,1])   #reshapes data to [2,3,1]
# print(tensor2)
# tf.Tensor(
# [[[1.]
#   [1.]
#   [1.]]

#  [[1.]
#   [1.]
#   [1.]]], shape=(2, 3, 1), dtype=float32)
tensor3 = tf.reshape(tensor2, [3,-1])   #-1 tells the tensor to calculate the size of the dimension in that place, this will reshape the tensor to [3,2]
# print(tensor3)
# tf.Tensor(
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]], shape=(3, 2), dtype=float32)
# The number of elements in the reshaped tensor MUST match the number in the original

# ex
t = tf.zeros([5,5,5,5])
# print(t)
t = tf.reshape(t, [625])
# print(t)
t = tf.reshape(t, [125,-1])
# print(t)













































