import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# digging into the data

# training data
print(f"Training data axes: {train_images.ndim}")
print(f"Training data shape: {train_images.shape}")
print(f"Training data shape: {train_images.dtype}")

# visualize an element in the training dataset
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

from tensorflow import keras 
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255 
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

with tf.device('/GPU:0'):
    model.fit(train_images, train_labels, epochs=20, batch_size=256)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
