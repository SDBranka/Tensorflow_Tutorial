# Deep Computer Vision
# In this guide we will learn how to peform image classification and 
# object detection/recognition using deep computer vision with something 
# called a convolutional neural network.

# The goal of our convolutional neural networks will be to classify and 
# detect images or specific objects from within the image. We will be 
# using image data as our features and a label for those images as our 
# label or output.

# We already know how neural networks work so we can skip through the 
# basics and move right into explaining the following concepts.

# Image Data
# Convolutional Layer
# Pooling Layer
# CNN Architectures
# The major differences we are about to see in these types of neural 
# networks are the layers that make them up.

# Image Data
# So far, we have dealt with pretty straight forward data that has 1 or
# 2 dimensions. Now we are about to deal with image data that is 
# usually made up of 3 dimensions. These 3 dimensions are as follows:

# image height
# image width
# color channels
# The only item in the list above you may not understand is color 
# channels. The number of color channels represents the depth of an image 
# and coorelates to the colors used in it. For example, an image with 
# three channels is likely made up of rgb (red, green, blue) pixels. So, 
# for each pixel we have three numeric values in the range 0-255 that 
# define its color. For an image of color depth 1 we would likely have a
# greyscale image with one value defining each pixel, again in the range 
# of 0-255. (see mod5_notes_pic1)Keep this in mind as we discuss how our 
# network works and the input/output of each layer.


# Convolutional Neural Network
# Note: I will use the term convnet and convolutional neural network 
# interchangably.
# Each convolutional neural network is made up of one or many 
# convolutional layers. These layers are different than the dense layers 
# we have seen previously. Their goal is to find patterns from within 
# images that can be used to classify the image or parts of it. But this 
# may sound familiar to what our densly connected neural network in the 
# previous section was doing, well that's becasue it is.
# The fundemental difference between a dense layer and a convolutional 
# layer is that dense layers detect patterns globally while convolutional 
# layers detect patterns locally. When we have a densly connected layer 
# each node in that layer sees all the data from the previous layer. 
# This means that this layer is looking at all the information and is 
# only capable of analyzing the data in a global capacity. Our 
# convolutional layer however will not be densly connected, this means 
# it can detect local patterns using part of the input data to that layer.

# Let's have a look at how a densly connected layer would look at an image 
# vs how a convolutional layer would.

# This is our image; the goal of our network will be to determine whether 
# this image is a cat or not. (see mod5_notes_pic2)
# Dense Layer: A dense layer will consider the ENTIRE image. It will look 
# at all the pixels and use that information to generate some output.
# Convolutional Layer: The convolutional layer will look at specific 
# parts of the image. In this example let's say it analyzes the highlighted 
# parts below and detects patterns there. (see mod5_notes_pic3)
# Can you see why this might make these networks more useful?

# How They Work
# A dense neural network learns patterns that are present in one specific 
# area of an image. This means if a pattern that the network knows is 
# present in a different area of the image it will have to learn the 
# pattern again in that new area to be able to detect it.
# Let's use an example to better illustrate this.
# We'll consider that we have a dense neural network that has learned what 
# an eye looks like from a sample of dog images. (see mod5_notes_pic4)
# Let's say it's determined that an image is likely to be a dog if an eye 
# is present in the boxed off locations of the image above.
# Now let's flip the image. (see mod5_notes_pic5)
# Since our densly connected network has only recognized patterns globally 
# it will look where it thinks the eyes should be present. Clearly it 
# does not find them there and therefore would likely determine this 
# image is not a dog. Even though the pattern of the eyes is present, 
# it's just in a different location.
# Since convolutional layers learn and detect patterns from different 
# areas of the image, they don't have problems with the example we just 
# illustrated. They know what an eye looks like and by analyzing different 
# parts of the image can find where it is present.


# Multiple Convolutional Layers
# In our models it is quite common to have more than one convolutional 
# layer. Even the basic example we will use in this guide will be made 
# up of 3 convolutional layers. These layers work together by increasing 
# complexity and abstraction at each subsequent layer. The first layer 
# might be responsible for picking up edges and short lines, while the 
# second layer will take as input these lines and start forming shapes 
# or polygons. Finally, the last layer might take these shapes and 
# determine which combiantions make up a specific image.


# Feature Maps
# You may see me use the term feature map throughout this tutorial. 
# This term simply stands for a 3D tensor with two spacial axes (width 
# and height) and one depth axis. Our convolutional layers take feature 
# maps as their input and return a new feature map that reprsents the 
# prescence of spcific filters from the previous feature map. These are 
# what we call response maps.


# Layer Parameters
# A convolutional layer is defined by two key parameters.
# Filters
# A filter is a m x n pattern of pixels that we are looking for in an 
# image. The number of filters in a convolutional layer reprsents how 
# many patterns each layer is looking for and what the depth of our 
# response map will be. If we are looking for 32 different 
# patterns/filters than our output feature map (aka the response map) 
# will have a depth of 32. Each one of the 32 layers of depth will be a 
# matrix of some size containing values indicating if the filter was 
# present at that location or not.
# Here's a great illustration from the book "Deep Learning with Python" 
# by Francois Chollet (pg 124). (see mod5_notes_pic6)

# Sample Size
# This isn't really the best term to describe this, but each convolutional 
# layer is going to examine n x m blocks of pixels in each image. 
# Typically, we'll consider 3x3 or 5x5 blocks. In the example above we 
# use a 3x3 "sample size". This size will be the same as the size of our
# filter.

# Our layers work by sliding these filters of n x m pixels over every 
# possible position in our image and populating a new feature 
# map/response map indicating whether the filter is present at each 
# location.

# Borders and Padding
# The more mathematical of you may have realized that if we slide a 
# filter of let's say size 3x3 over our image well consider less positions 
# for our filter than pixels in our input. Look at the example below.

# Image from "Deep Learning with Python" by Francois Chollet (pg 126).
# (mod5_notes_pic7)


# This means our response map will have a slightly smaller width and 
# height than our original image. This is fine but sometimes we want our 
# response map to have the same dimensions. We can accomplish this by 
# using something called padding.

# Padding is simply the addition of the appropriate number of rows and/or 
# columns to your input data such that each pixel can be centered by 
# the filter.

# Strides
# In the previous sections we assumed that the filters would be slid 
# continously through the image such that it covered every possible 
# position. This is common but sometimes we introduce the idea of a 
# stride to our convolutional layer. The stride size reprsents how many 
# rows/cols we will move the filter each time. These are not used very 
# frequently so we'll move on.

# Pooling
# You may recall that our convnets are made up of a stack of convolution 
# and pooling layers.

# The idea behind a pooling layer is to downsample our feature maps and 
# reduce their dimensions. They work in a similar way to convolutional 
# layers where they extract windows from the feature map and return a 
# response map of the max, min or average values of each channel. Pooling 
# is usually done using windows of size 2x2 and a stride of 2. This will 
# reduce the size of the feature map by a factor of two and return a 
# response map that is 2x smaller.


# CNN Architecture
# A common architecture for a CNN is a stack of Conv2D and MaxPooling2D 
# layers followed by a few denesly connected layers. To idea is that the 
# stack of convolutional and maxPooling layers extract the features from 
# the image. Then these features are flattened and fed to densly connected 
# layers that determine the class of an image based on the presence of 
# features.


# Working with Small Datasets
# In the situation where you don't have millions of images it is 
# difficult to train a CNN from scratch that performs very well


# Data Augmentation
# To avoid overfitting and create a larger dataset from a smaller one we 
# can use a technique called data augmentation. This is simply performing 
# random transofrmations on our images so that our model can generalize 
# better. These transformations can be things like compressions, rotations, 
# stretches and even color changes.

# Fortunately, keras can help us do this.


# Pretrained Models
# You would have noticed that the model above takes a few minutes to 
# train in the NoteBook and only gives an accuaracy of ~70%. This is okay 
# but surely there is a way to improve on this.

# In this section we will talk about using a pretrained CNN as apart of 
# our own custom network to improve the accuracy of our model. We know 
# that CNN's alone (with no dense layers) don't do anything other than 
# map the presence of features from our input. This means we can use a 
# pretrained CNN, one trained on millions of images, as the start of our 
# model. This will allow us to have a very good convolutional base before 
# adding our own dense layered classifier at the end. In fact, by using 
# this techique we can train a very good classifier for a realtively small 
# dataset (< 10,000 images). This is because the convnet already has a 
# very good idea of what features to look for in an image and can find 
# them very effectively. So, if we can determine the presence of features 
# all the rest of the model needs to do is determine which combination of 
# features makes a specific image.

# Fine Tuning
# When we employ the technique defined above, we will often want to 
# tweak the final layers in our convolutional base to work better for our 
# specific problem. This involves not touching or retraining the earlier 
# layers in our convolutional base but only adjusting the final few. We 
# do this because the first layers in our base are very good at extracting 
# low level features lile lines and edges, things that are similar for
# any kind of image. Where the later layers are better at picking up very 
# specific features like shapes or even eyes. If we adjust the final 
# layers than we can look for only features relevant to our very specific 
# problem.

# Using a Pretrained Model
# In this section we will combine the tecniques we learned above and use 
# a pretrained model and fine tuning to classify images of dogs and cats 
# using a small dataset.



