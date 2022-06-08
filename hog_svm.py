import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from keras_HOG.hog import HOG
import numpy as np
import matplotlib.pyplot as plt


# physical_devices = tf.config.list_physical_devices('GPU') 
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

def plot_learning_curve(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    fig0= plt.figure(0)
    plt.xlabel("Epochs: ")
    plt.ylabel("Loss: ")
    plt.plot(range(len(train_loss)), train_loss)
    plt.plot(range(len(val_loss)), val_loss)
    plt.show()
    plt.pause(5)
    fig0.savefig('Loss.png')
    

    fig1= plt.figure(1)
    plt.xlabel("Epochs: ")
    plt.ylabel("accuracy: ")
    plt.plot(range(len(train_acc)), train_acc)
    plt.plot(range(len(val_acc)), val_acc)
    plt.show()
    fig1.savefig('Accuracy.png')
    
NUM_CLASSES = 6
RESULT_DIR = '/home/ducanh/project_master/final_project_cv/converted_dataset'

datagenerator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
    validation_split=0.2,
)

train_ds = datagenerator.flow_from_directory(
    directory = RESULT_DIR,
    class_mode='categorical',
    color_mode='rgb',
    batch_size=16,
    target_size=(180, 360),
    shuffle=True,
    seed=None,
    subset='training',
    interpolation='bilinear',
    follow_links=False,
)

# train_ds = tf.keras.image_dataset_from_directory(
#     directory = RESULT_DIR,
#     labels='inferred',
#     label_mode='int',
#     class_names=None,
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(240, 320),
#     shuffle=True,
#     seed=None,
#     validation_split=0.2,
#     subset='training',
#     interpolation='bilinear',
#     follow_links=False,
#     crop_to_aspect_ratio=True,
# )
 


val_ds = datagenerator.flow_from_directory(
    directory = RESULT_DIR,
    class_mode='categorical',
    color_mode='rgb',
    batch_size=16,
    target_size=(180, 360),
    shuffle=True,
    seed=None,
    subset='validation',
    interpolation='bilinear',
    follow_links=False,
)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     directory = RESULT_DIR,
#     labels='inferred',
#     label_mode='int',
#     class_names=None,
#     color_mode='rgb',
#     batch_size=32,
#     image_size=(240, 320),
#     shuffle=True,
#     seed=None,
#     validation_split=0.2,
#     subset='validation',
#     interpolation='bilinear',
#     follow_links=False,
#     crop_to_aspect_ratio=True,
# )


hog = HOG()
svm =  keras.Sequential(
    [
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=NUM_CLASSES),
    ], name='svm')

hog_svm = keras.models.Sequential([
    hog,
    svm
])

hog_svm.build(input_shape=(None, 180, 360, 3))
hog_svm.summary()

hog_svm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

epochs=50
history = hog_svm.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



hog_svm.save('model_final_3105')
plot_learning_curve(history)
