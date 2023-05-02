import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image                            # for data augmentation
from keras.preprocessing.image import ImageDataGenerator         # for data augmentation


# # Data Augmentation
# # creates a data generator object that transforms images
# datagen = ImageDataGenerator(
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest')

# # pick an image to transform
# test_img = train_images[20]
# # convert image to numpy array
# img = image.img_to_array(test_img)  
# # reshape image
# img = img.reshape((1,) + img.shape)  

# i = 0

# # this loops runs forever until we break, saving images to current directory with specified prefix
# for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  
#     plt.figure(i)
#     plot = plt.imshow(image.img_to_array(batch[0]))
#     i += 1
#     # show 4 images
#     if i > 4:  
#         break

# plt.show()
