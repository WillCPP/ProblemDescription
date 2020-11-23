from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

class DNN():
    def __init__(self, input_shape, num_classes):
        self.model = None
        self.input_shape = input_shape
        self.num_classes = num_classes

    def createModel(self):
        layers = [
            keras.Input(shape=self.input_shape),
            Conv2D(32, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(self.num_classes, activation="softmax"),
        ]
        model = Sequential(layers)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()])
        self.model = model

    def train(self, x_train, y_train, x_test, y_test):
        if self.model is None:
            print('Model has not been created yet, run createModel() first.')
            return
        else:
            batch_size = 128
            epochs = 15
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = keras.models.load_model(path)
