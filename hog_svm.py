import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras_HOG.hog import HOG

import cv2
import numpy as np

NUM_CLASSES = 6

hog = HOG()
svm = model = keras.Sequential(
    [
        keras.layers.Input(shape=(40716,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=NUM_CLASSES),
    ], name='svm')

raw_image = cv2.imread('9.jpeg')
with open('./9.txt', 'r') as f:
    coords = np.array([line.strip().split(' ') for line in f.readlines()]).astype(np.float32)

x1, y1, x2, y2 = [int(c) for c in [coords[0][0],coords[0][1],coords[0][2],coords[0][3]]]

input_img = raw_image[y1:y2, x1:x2, :]
input_img = input_img[None, :, :, :]
input_img = tf.image.resize(input_img, (240, 320))
print(input_img.shape)
# cv2.imshow("Window", mat=input_img)
# cv2.waitKey(0)

hog_feat = hog(input_img)
print(hog_feat.shape)

hog_svm = keras.models.Sequential([
    hog,
    svm
])

hog_svm.build(input_shape=(None, 240, 320, 3))
hog_svm.summary()