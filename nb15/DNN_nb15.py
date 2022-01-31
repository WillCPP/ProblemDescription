import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow_privacy.privacy.keras_models.dp_keras_model import DPSequential
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DNN():
    def __init__(self, input_shape, num_classes, params, use_tf_privacy=False, noise_multiplier=None):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.params = params
        self.use_tf_privacy = use_tf_privacy
        self.noise_multiplier = noise_multiplier

    def createModel(self):
        # layers = [
        #     keras.Input(shape=self.input_shape),
        #     Conv1D(64, kernel_size=(1), activation="relu", padding="same"),
        #     Conv1D(64, kernel_size=(1), activation="relu", padding="same"),
        #     MaxPooling1D(pool_size=(2)),
        #     Conv1D(32, kernel_size=(1), activation="relu", padding="same"),
        #     Conv1D(32, kernel_size=(1), activation="relu", padding="same"),
        #     MaxPooling1D(pool_size=(2)),
        #     Flatten(),
        #     Dense(256),
        #     Activation("relu"),
        #     Dropout(0.5),
        #     Dense(self.num_classes, activation="softmax"),
        # ]
        layers = [
            keras.Input(shape=self.input_shape),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax'),
        ]
        model = Sequential(layers)
        model.summary()

        if self.use_tf_privacy:
            loss = loss=keras.losses.CategoricalCrossentropy(from_logits=True)#, reduction=tf.losses.Reduction.NONE)
        else:
            loss = loss=keras.losses.CategoricalCrossentropy(from_logits=True)

        model.compile(optimizer=keras.optimizers.Adam(), loss=loss, metrics=[keras.metrics.CategoricalAccuracy()])
        self.model = model

    def train(self, x_train, y_train, x_test, y_test):
        if self.model is None:
            print('Model has not been created yet, run createModel() first.')
            return
        else:
            optimizer = None
            if self.use_tf_privacy:
                optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=self.params['l2_norm_clip'],
                noise_multiplier=self.noise_multiplier,
                learning_rate=self.params['learning_rate'],
                num_microbatches=self.params['num_microbatches'])
            else:
                # optimizer=keras.optimizers.Adam(0.001)
                optimizer=keras.optimizers.Adam(self.params['learning_rate'])

            self.model.compile(optimizer=optimizer,
                loss=keras.losses.CategoricalCrossentropy(from_logits=True), # TODO: Check if from_logits=True is needed here
                metrics=[keras.metrics.CategoricalAccuracy()])

            self.model.fit(x_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], validation_split=self.params['validation_split'])

            # self.model.fit(x_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'])
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
