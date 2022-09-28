import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, UpSampling2D, UpSampling3D, Add, BatchNormalization, Input, Activation, Lambda, Concatenate
from GAN_models import Generator_Pix2Pix_Concat, Generator_Pix2Pix_Add, Generator_ResNet_ED_Concat, Generator_ResNet_ED_Add, Generator_Xception_ED_Concat, Generator_Xception_ED_Add, Discriminator_Pix2Pix, Discriminator_ResNet, Discriminator_Xception

IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

seg_path = 'D:/NYUV2 GAN/imgs_seg/'
image_path = 'D:/NYUV2 GAN/images/'

file = '1'

input_image = tf.io.read_file(seg_path+file+'.png')
input_image = tf.image.decode_png(input_image, channels=1)

target_image = tf.io.read_file(image_path+file+'.png')
target_image = tf.image.decode_png(target_image, channels=3)

input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
target_image = tf.image.resize(target_image, [IMG_HEIGHT, IMG_WIDTH],method=tf.image.ResizeMethod.BILINEAR)

input_image = tf.expand_dims(input_image, axis=0)
input_image = tf.cast(input_image, dtype=tf.float32)
#target_image = tf.cast(target_image, dtype=tf.float32)

model = Generator_ResNet_ED_Add()
model.load_weights('weights.h5')
model.summary()


pred = model.predict(input_image)
pred[pred>1] = 1
pred[pred<-1] = -1
pred = np.uint8((pred[0]+1)*127.5)
target_image = np.uint8(target_image)
plt.figure('Generated Image')
plt.imshow(pred)
plt.figure('GT')
plt.imshow(target_image)
plt.show()