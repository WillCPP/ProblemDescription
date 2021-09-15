from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
import numpy as np
from DNN import DNN
from parameters_sb import parameters
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

def base_model_sb(save_model=False, model_folder=None, use_tf_privacy=False, noise_multiplier=None):
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1,)
    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Scale images to the [0, 1] range
    x_train = np.array(x_train, dtype=np.float32) / 255  # x_train.astype("float32") / 255
    x_test = np.array(x_test, dtype=np.float32) / 255  # x_test.astype("float32") / 255
    
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if not use_tf_privacy:
        dnn = DNN(input_shape, num_classes, parameters['base_model'])
        dnn.createModel()
        dnn.train(x_train, y_train, x_test, y_test)
        if save_model: dnn.saveModel(model_folder + '/model_plain')
        score = dnn.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        return dnn.model

    if use_tf_privacy:
        dnn = DNN(input_shape, num_classes, parameters['base_model'], True, noise_multiplier)
        dnn.createModel()
        dnn.train(x_train, y_train, x_test, y_test)
        if save_model: dnn.saveModel(model_folder + '/model_private')
        score = dnn.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        compute_dp_sgd_privacy.compute_dp_sgd_privacy(x_train.shape[0], \
                                                        parameters['base_model']['batch_size'], \
                                                        noise_multiplier, \
                                                        parameters['base_model']['epochs'],
                                                        1e-5)
        return dnn.model