# Data Augmentation


import numpy as np
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


# creates a data generator object that transforms images
datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest'
                            )

# pick an image to transform
test_img = train_images[20]
# convert image to numpy array
# img = image.img_to_array(test_img)  
img = tf.keras.utils.array_to_img(test_img)
array = tf.keras.utils.image.img_to_array(img)
# reshape image
img = img.reshape((1,) + img.shape)  

i = 0

# this loops runs forever until we break, saving images to current directory with specified prefix
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  
    plt.figure(i)
    # plot = plt.imshow(image.img_to_array(batch[0]))
    plot = plt.imshow(tf.keras.utils.image.img_to_array(batch[0]))
    i += 1
    # show 4 images
    if i > 4:  
        break

plt.show()





# # Building the Convolutional Base
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), 
#                         activation='relu', 
#                         input_shape=(32, 32, 3)
#                         ))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# # Adding Dense Layers
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))


# # compile and train the model using the recommended hyper paramaters 
# # from tensorflow
# model.compile(optimizer='adam',
#             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy']
#             )
# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))




