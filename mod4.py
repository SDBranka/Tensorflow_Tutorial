# Use a neural network to classify articles of clothing
# To achieve this, we will use a sub module of TensorFlow called keras.


# keras
# "Keras is a high-level neural networks API, written in Python and 
# capable of running on top of TensorFlow, CNTK, or Theano. It was 
# developed with a focus on enabling fast experimentation.

# Use Keras if you need a deep learning library that:

# Allows for easy and fast prototyping (through user friendliness, 
# modularity, and extensibility).
# Supports both convolutional networks and recurrent networks, as well 
# as combinations of the two.
# Runs seamlessly on CPU and GPU."
# Keras is a very powerful module that allows us to avoid having to build 
# neural networks from scratch. It also hides a lot of mathematical 
# complexity (that otherwise we would have to implement) inside of 
# helpful packages, modules and methods.


# Neural Networks
# On a lower level neural networks are simply a combination of elementry 
# math operations and some more advanced linear algebra
# Each neural network consists of a sequence of layers in which data 
# passes through. These layers are made up on neurons and the neurons 
# of one layer are connected to the next (see below). These connections
# are defined by what we call a weight (some numeric value). Each layer 
# also has something called a bias, this is simply an extra neuron that 
# has no connections and holds a single numeric value. Data starts at 
# the input layer and is trasnformed as it passes through subsequent 
# layers. The data at each subsequent neuron is defined as the following.

# Y=(∑ni=0wixi)+b 

# w  stands for the weight of each connection to the neuron
# x  stands for the value of the connected neuron from the previous value
# b  stands for the bias at each layer, this is a constant
# n  is the number of connections
# Y  is the output of the current neuron
# ∑  stands for sum

# The equation you just read is called a weighed sum. We will take this 
# weighted sum at each and every neuron as we pass information through 
# the network. Then we will add what's called a bias to this sum. The 
# bias allows us to shift the network up or down by a constant value. 
# It is like the y-intercept of a line.

# But that equation is the not complete one! We forgot a crucial part, 
# the activation function. This is a function that we apply to the 
# equation seen above to add complexity and dimensionality to our network. 
# Our new equation with the addition of an activation function  F(x)  
# is seen below.

# Y=F((∑ni=0wixi)+b)

# Our network will start with predefined activation functions (they may 
# be different at each layer) but random weights and biases. As we train 
# the network by feeding it data it will learn the correct weights and 
# biases and adjust the network accordingly using a technqiue called 
# backpropagation. Once the correct weights and biases have been learned 
# our network will hopefully be able to give us meaningful predictions. 
# We get these predictions by observing the values at our final layer, 
# the output layer.













































































