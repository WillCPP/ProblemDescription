import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, MaxPool2D
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
        #     Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Conv2D(128, kernel_size=(3, 3), activation="relu"),
        #     MaxPooling2D(pool_size=(2, 2)),
        #     Flatten(),
        #     Dropout(0.5),
        #     Dense(self.num_classes, activation="softmax"),
        # ]
        layers = [
            keras.Input(shape=self.input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256),
            Activation("relu"),
            Dropout(0.5),
            Dense(self.num_classes, activation="softmax"),
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

            # self.model.fit(x_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], validation_split=self.params['validation_split'])

            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # dimesion reduction
                rotation_range=0.1,  # randomly rotate images in the range
                zoom_range = 0.1, # Randomly zoom image
                width_shift_range=0.1,  # randomly shift images horizontally
                height_shift_range=0.1,  # randomly shift images vertically
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            self.model.fit(datagen.flow(x_train, y_train), batch_size=self.params['batch_size'], epochs=self.params['epochs'])
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
