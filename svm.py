import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures


model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data by flattening & scaling it
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_test.reshape(-1, 784).astype("float32") / 255

# Categorical (one hot) encoding of the labels
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)