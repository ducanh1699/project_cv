from curses import window
import tensorflow as tf
import tensorflow.keras as keras
from skimage.transform import pyramid_gaussian
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from skimage import color
from keras_HOG.hog import HOG
import os
import cv2
import numpy as np
from slidingwindow import sliding_window
from imutils.object_detection import non_max_suppression

# def your_softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0) # only difference

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

load_model = keras.models.load_model('model_final_3105')
# load_model.summarry()
# print(load_model.output_shape)
img = cv2.imread('14.jpeg')
# load_model.predict((img).shape)
w = 1280 
h = 720
dim = (h,w)
# image = tf.keras.preprocessing.image.load_img('9.jpeg', target_size = dim )
# input_arr = tf.keras.preprocessing.image.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.
# predictions = load_model.predict(input_arr)
# print(predictions)
# predictions = np.array(predictions)
# output = softmax(predictions)
# output = tf.keras.activations.softmax(
#     predictions[0], axis=-1
# )
# print(output)
# resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
# height = img.shape[0]
# width = img.shape[1]
# image_size = np.array([width, height])
# resize_img = tf.keras.layers.Reshape(resize_img)
# load_model.predict(resize_img, batch_size =None)
# print(resize_img.shape)
# print(resize_img)

(winW, winH)= (360,180)
windowSize=(winW,winH)
downscale=2
scale = 0
detections = []
for resize in pyramid_gaussian(image=img, downscale=2, multichannel = True):
    print(resize.shape)
    for (x,y,window) in sliding_window(resize, stepSize = 20, windowSize=(winW,winH)):
        if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met the minimum size requirement
            continue
        image = tf.expand_dims(window, axis=0)
        # input_arr = tf.keras.preprocessing.image.img_to_array(image)
        # input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = load_model.predict(image)
        # print(predictions)
        # predictions = np.array(predictions)
        output = softmax(predictions)
        # print(output)
        # window = color.rgb2gray(window)
        # print(window)
        # loss, = load_model(window)
        if output.max() > 0.9:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} | Class {}\n".format(scale, output.max(), np.argmax(output)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), output.max(),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
       
    scale+=1


# for (x_tl, y_tl, _, w, h) in detections:
#     cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score for (x, y, score, w, h) in detections]
print("detection confidence score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)

# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
cv2.imshow("Raw Detections after NMS", img)
#### Save the images below
k = cv2.waitKey(0) & 0xFF 
if k == 27:             #wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('saved_image.png',img)
    cv2.destroyAllWindows()