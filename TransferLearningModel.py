from DNN import DNN
from tensorflow import keras
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TransferLearningModel:
    def __init__(self, params, use_tf_privacy=False, noise_multiplier=None):
        self.model = None
        self.params = params
        self.use_tf_privacy = use_tf_privacy
        self.noise_multiplier = noise_multiplier
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        if not self.use_tf_privacy:
            self.model = keras.models.load_model(path)
        if self.use_tf_privacy:
            # optimizer = DPKerasAdamOptimizer(
            #     l2_norm_clip=self.params['l2_norm_clip'], 
            #     noise_multiplier=self.noise_multiplier,
            #     learning_rate=self.params['learning_rate'])
            # self.model = keras.models.load_model(path, compile=False)  #, custom_objects={'DPOptimizerClass': optimizer})
            # self.model.compile(optimizer=optimizer,
            #     loss=keras.losses.CategoricalCrossentropy(),
            #     metrics=[keras.metrics.CategoricalAccuracy()])
            # ==================================================
            dnn = DNN()
            dnn.createModel()
            dnn.model.set_weights(keras.models.load_model(path).get_weights())
            self.model = dnn.model

    def fineTune(self, x_train, y_train):
        for i in range(len(self.model.layers) - self.params['num_layers']):
            self.model.layers[i].trainable = False

        self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=[keras.metrics.CategoricalAccuracy()]) # TODO: This line is probably needed

        for i in range(len(self.model.layers)):
            print(f'Layer: {i} | Trainable: {self.model.layers[i].trainable}')
        self.model.summary()
        
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
            loss=keras.losses.CategoricalCrossentropy(),
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

        weights_list_after = []
        for layer in self.model.layers:
            weights_list_after.append(layer.get_weights())