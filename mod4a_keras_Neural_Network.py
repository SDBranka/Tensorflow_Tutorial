# Use a neural network to classify articles of clothing
# To achieve this, we will use a sub module of TensorFlow called keras.

# For this tutorial we will use the MNIST Fashion Dataset. This is a 
# dataset that is included in keras.
# This dataset includes 60,000 images for training and 10,000 images 
# for validation/testing.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# load dataset
fashion_mnist = keras.datasets.fashion_mnist  
# split into testing and training
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  

# # take a look at the data
# print(train_images.shape)
# # (60000, 28, 28)
# # this translates as 60,000 images that are made up of 28x28 pixels
# print(train_images[0])
# # this is a numpy.ndarray
# # let's have a look at one pixel
# print(train_images[0,23,23])
# # 194
# # Our pixel values are between 0 and 255, 0 being black and 255 being 
# # white. This means we have a grayscale image as there are no color
# # channels.

# # show the training labels, the values will be between 0 and 9 because
# # there are 10 training labels
# print(train_labels[:10])
# # [9 0 0 3 0 2 7 2 5 5]


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # chart1
# # let's look at what some of these images look like
# plt.figure()
# plt.imshow(train_images[1])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# In neural networks all the values must be between 0 and 1
# The last step before creating our model is to preprocess our data. 
# This simply means applying some prior transformations to our data 
# before feeding it the model. In this case we will simply scale all 
# our greyscale pixel values (0-255) to be between 0 and 1. We can do 
# this by dividing each value in the training and testing sets by 255.0.
# We do this because smaller values will make it easier for the model to
# process our values.
train_images = train_images / 255.0
test_images = test_images / 255.0


# Building the Model
# Use a keras sequential model with three different layers. This model 
# represents a feed-forward neural network (one that passes values from 
# left to right)
model = keras.Sequential([
    # input layer (1)
    # it will conist of 784 neurons. We use the flatten layer with an 
    # input shape of (28,28) to denote that our input should come in that 
    # shape. The flatten means that our layer will reshape the shape (28,28)
    # array into a vector of 784 neurons so that each pixel will be 
    # associated with one neuron
    keras.layers.Flatten(input_shape=(28, 28)),  
    # hidden layer (2)
    # Our first and only hidden layer. The dense denotes that this layer 
    # will be fully connected and each neuron from the previous layer 
    # connects to each neuron of this layer. It has 128 neurons and uses 
    # the rectify linear unit activation function (mod4_notes_pic1).
    keras.layers.Dense(128, activation='relu'),  
    # output layer (3)
    # has 10 neurons that we will look at to determine our models output. 
    # Each neuron represnts the probabillity of a given image being one of 
    # the 10 different classes. The activation function softmax is used on 
    # this layer to calculate a probabillity distribution for each class. 
    # This means the value of any neuron in this layer will be between 0 
    # and 1, where 1 represents a high probabillity of the image being that 
    # class.
    keras.layers.Dense(10, activation='softmax') 
])


# Compile the Model
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )


# Training the Model
# pass the data, labels and epochs
model.fit(train_images, train_labels, epochs=10) 


# Evaluating the Model
# The verbose argument is defined from the keras documentation as:
# "verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 

# print('Test accuracy:', test_acc)
# # Test accuracy: 0.8845999836921692


# Making Predictions
# To make predictions we simply need to pass an array of data in the form 
# we've specified in the input layer to .predict() method.
predictions = model.predict(test_images)

# # This method returns to us an array of predictions for each image we 
# # passed it. Let's have a look at the predictions for image 1
# print(predictions[0])
# # [4.7735584e-09 1.4110267e-12 1.6301328e-09 5.1225479e-10 5.5781545e-08
# #  1.0809021e-03 3.7694832e-08 6.0817068e-03 6.5559612e-07 9.9283653e-01]

# # If we wan't to get the value with the highest score we can use a 
# # useful function from numpy called argmax(). This simply returns the 
# # index of the maximium value from a numpy array.
# print(np.argmax(predictions[0]))
# # 9

# # we can check if this is correct by looking at the value of the 
# # cooresponding test label
# print(test_labels[0])
# # 9



# Verifying Predictions
# a small function here to help us verify predictions with some simple visuals.
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def get_number():
    while True:
        num = input("Pick a number: ")
        # entered '9' recieved chart2
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)










