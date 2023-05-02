import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# Dataset
# We will load the cats_vs_dogs dataset from the modoule 
# tensorflow_datatsets.
# This dataset contains (image, label) pairs where images have different 
# dimensions and 3 color channels.

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs',
                                                            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                            with_info=True,
                                                            as_supervised=True,
                                                            )


# creates a function object that we can use to get labels
get_label_name = metadata.features['label'].int2str 

# display 2 images from the dataset
for image, label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


# Data Preprocessing
# Since the sizes of our images are all different, we need to convert 
# them all to the same size. We can create a function that will do that 
# for us below.
IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
    # returns an image that is reshaped to IMG_SIZE
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# Now we can apply this function to all our images using .map().
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)


# Let's have a look at our images now.
for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


# Finally we will shuffle and batch the images.
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# # Now if we look at the shape of an original image vs the new image we 
# # will see it has been changed.
# for img, label in raw_train.take(2):
#     print("Original shape:", img.shape)
# # Original shape: (262, 350, 3)
# # Original shape: (409, 336, 3)


# for img, label in train.take(2):
#     print("New shape:", img.shape)
# # New shape: (160, 160, 3)
# # New shape: (160, 160, 3)


# Picking a Pretrained Model
# The model we are going to use as the convolutional base for our model 
# is the MobileNet V2 developed at Google. This model is trained on 1.4 
# million images and has 1000 different classes.

# We want to use this model but only its convolutional base. So, when we 
# load in the model, we'll specify that we don't want to load the top 
# (classification) layer. We'll tell the model what input shape to 
# expect and to use the predetermined weights from imagenet (Googles 
# dataset).
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet'
                                                )
# base_model.summary()


# At this point this base_model will simply output a shape 
# (32, 5, 5, 1280) tensor that is a feature extraction from our original 
# (1, 160, 160, 3) image. The 32 means that we have 32 layers of 
# different filters/features.
for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
# print(feature_batch.shape)
# # (32, 5, 5, 1280)



























