# The problem we will consider here is classifying 10 different everyday 
# objects. The dataset we will use is built into tensorflow and called 
# the CIFAR Image Dataset. It contains 60,000 32x32 color images with 
# 6000 images of each class.


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image                            # for data augmentation
from keras.preprocessing.image import ImageDataGenerator         # for data augmentation


#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
                ]


# # chart1
# # Let's look at a one image
# IMG_INDEX = 7  # change this to look at other images
# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()


# Building the Convolutional Base
model = models.Sequential()
# after each convolutional layer we have some sort of maxpooling layer
# typically to reduce the dimensionality
# in our first layer we define 32 filters with a sampling size of 3x3
# using the relu activation function with an input shape of 32x32x3
model.add(layers.Conv2D(32, (3, 3), 
                        activation='relu', 
                        input_shape=(32, 32, 3)
                        ))
model.add(layers.MaxPooling2D((2, 2)))
# we don't need to define the input shape in these subsequent layers as 
# they are determined by the previous layers
# this second convolutional layer uses 64 filters with a sampling size
# of 3x3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Layer 1
# The input shape of our data will be 32, 32, 3 and we will process 32 
# filters of size 3x3 over our input data. We will also apply the 
# activation function relu to the output of each convolution operation.
# Layer 2
# This layer will perform the max pooling operation using 2x2 samples 
# and a stride of 2.
# Other Layers
# The next set of layers do very similar things but take as input the 
# feature map from the previous layer. They also increase the frequency 
# of filters from 32 to 64. We can do this as our data shrinks in 
# spacial dimensions as it passed through the layers, meaning we can
# afford (computationally) to add more depth.


# # take a look at the summary
# print(model.summary())
# # Model: "sequential"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #
# # =================================================================
# #  conv2d (Conv2D)             (None, 30, 30, 32)        896

# #  max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
# #  )

# #  conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496

# #  max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
# #  2D)

# #  conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928

# # =================================================================
# # Total params: 56,320
# # Trainable params: 56,320
# # Non-trainable params: 0
# # _________________________________________________________________
# # None

# Note that the output shape of the initial layer is 30x30x32 instead of 
# 32x32x32, this is because we haven't added padding
# Note that the max pooling layer halves the size


# Adding Dense Layers
# Now we need to take these extracted features and add a way to 
# classify them
# first we flatten the data to a single vector
model.add(layers.Flatten())
# next create a 64 neuron dense layer
model.add(layers.Dense(64, activation='relu'))
# lastly create an output layer of 10 neurons which corresponds to the 
# number of classification categories
model.add(layers.Dense(10))

# # look again at the model summary
# print(model.summary())
# # Model: "sequential"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #
# # =================================================================
# #  conv2d (Conv2D)             (None, 30, 30, 32)        896

# #  max_pooling2d (MaxPooling2D  (None, 15, 15, 32)       0
# #  )

# #  conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496

# #  max_pooling2d_1 (MaxPooling  (None, 6, 6, 64)         0
# #  2D)

# #  conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928

# #  flatten (Flatten)           (None, 1024)              0

# #  dense (Dense)               (None, 64)                65600

# #  dense_1 (Dense)             (None, 10)                650

# # =================================================================
# # Total params: 122,570
# # Trainable params: 122,570
# # Non-trainable params: 0
# # _________________________________________________________________
# # None

# We can see that the flatten layer changes the shape of our data so 
# that we can feed it to the 64-node dense layer, followed by the final 
# output layer of 10 neurons (one for each class).


# train and compile the model using the recommended hyper paramaters 
# from tensorflow
model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
            )

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# # Evaluating the Model
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

# # view the test accuracy
# print(test_acc)
# # 0.7042999863624573

# # so about 70% accuracy


# Data Augmentation
# creates a data generator object that transforms images
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# pick an image to transform
test_img = train_images[20]
# convert image to numpy array
img = image.img_to_array(test_img)  
# reshape image
img = img.reshape((1,) + img.shape)  

i = 0

# this loops runs forever until we break, saving images to current directory with specified prefix
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    # show 4 images
    if i > 4:  
        break

plt.show()



















