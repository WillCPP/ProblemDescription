from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
import numpy as np
from TransferLearningModel import TransferLearningModel
from DatasetSampler import DatasetSampler
from parameters import parameters

def transfer_model_emnist(model_folder, use_tf_privacy=False, noise_multiplier=None):
    # Model / data parameters
    num_classes = 10

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

    if not use_tf_privacy:
        tlm = TransferLearningModel(parameters['transfer_model_emnist'])
        tlm.loadModel(model_folder + '/model_plain')
        tlm.model.summary()
        print(tlm.model.layers[0].trainable)
        tlm.fineTune(x_train, y_train)
        score = tlm.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

    if use_tf_privacy:
        tlm_private = TransferLearningModel(parameters['transfer_model_emnist'], True, noise_multiplier)
        tlm_private.loadModel(model_folder + '/model_private')
        tlm_private.model.summary()
        print(tlm_private.model.layers[0].trainable)
        tlm_private.fineTune(x_train, y_train)
        score = tlm_private.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])