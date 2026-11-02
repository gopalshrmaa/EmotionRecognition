import tensorflow as tf
from tensorflow import keras
import cv2

print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)

# Create a tiny test CNN
model = keras.Sequential([
    keras.layers.Input(shape=(48, 48, 1)),
    keras.layers.Conv2D(16, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(7, activation="softmax")
])

model.summary()
print("CNN created successfully!")