from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
import tensorflow_datasets as tfds
from TransferLearningModel import TransferLearningModel
from DatasetSampler import DatasetSampler

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Sampling for 20 instances per class
ds = DatasetSampler()
x_train, y_train = ds.sample(x_train, y_train, 20)
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tlm = TransferLearningModel()
tlm.loadModel('model_0')
tlm.model.summary()
print(tlm.model.layers[0].trainable)
tlm.fineTune(x_train, y_train, 5)
score = tlm.model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])