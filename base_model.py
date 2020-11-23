from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from DNN import DNN

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

dnn = DNN(input_shape, num_classes)
dnn.createModel()
dnn.train(x_train, y_train, x_test, y_test)
dnn.saveModel('model_0')
score = dnn.model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# dnn2 = DNN(input_shape, num_classes)
# dnn2.loadModel('model_0')
# score2 = dnn.model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score2[0])
# print("Test accuracy:", score2[1])