from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from TransferLearningModel import TransferLearningModel
from DatasetSampler import DatasetSampler
from parameters_sb import parameters

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# emnist_train = tfds.load('emnist/mnist', split='train', batch_size=-1)  # tfds.as_numpy(emnist_train) #as_supervised=True,
# emnist_test = tfds.load('emnist/mnist', split='test', batch_size=-1)  # tfds.as_numpy(emnist_test) #as_supervised=True,
# emnist_train = tfds.as_numpy(emnist_train)
# emnist_test = tfds.as_numpy(emnist_test)
# print(type(emnist_train['image']))
# (x_train, y_train) = emnist_train['image'], emnist_train['label']  # emnist_train[0], emnist_train[1]  
# (x_test, y_test) =  emnist_test['image'], emnist_test['label']  # emnist_test[0], emnist_test[1]

# Load transfer training dataset
x_train = np.load('data/transfer_training_dataset/x_train.npy')
y_train = np.load('data/transfer_training_dataset/y_train.npy')
x_test = np.load('data/transfer_training_dataset/x_test.npy')
y_test = np.load('data/transfer_training_dataset/y_test.npy')

# Sampling for 20 instances per class
ds = DatasetSampler()
x_train, y_train = ds.sample(x_train, y_train, 20)
x_test, y_test= ds.sample(x_test, y_test, 20)
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
tlm.loadModel('models/model_0')
tlm.model.summary()
print(tlm.model.layers[0].trainable)
tlm.fineTune(x_train, y_train, parameters['transfer_model_emnist'], False)
score = tlm.model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

tlm_private = TransferLearningModel()
tlm_private.loadModel('models/model_0')
tlm_private.model.summary()
print(tlm_private.model.layers[0].trainable)
tlm_private.fineTune(x_train, y_train, parameters['transfer_model_emnist'], True)
score = tlm_private.model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])