from tensorflow import keras
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

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
            optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=self.params['l2_norm_clip'], 
                noise_multiplier=self.noise_multiplier,
                learning_rate=self.params['learning_rate'])
            self.model = keras.models.load_model(path, compile=False)  #, custom_objects={'DPOptimizerClass': optimizer})
            self.model.compile(optimizer=optimizer,
                loss=keras.losses.CategoricalCrossentropy(),
                metrics=[keras.metrics.CategoricalAccuracy()])

    def fineTune(self, x_train, y_train):
        for i in range(len(self.model.layers) - self.params['num_layers']):
            self.model.layers[i].trainable = False
        
        optimizer = None
        if self.use_tf_privacy:
            optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=self.params['l2_norm_clip'], 
            noise_multiplier=self.noise_multiplier,
            learning_rate=self.params['learning_rate'])
        else:
            optimizer=keras.optimizers.Adam(0.001)
        
        self.model.compile(optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()])
        
        self.model.fit(x_train, y_train, batch_size=self.params['batch_size'], epochs=self.params['epochs'], validation_split=self.params['validation_split'])

        weights_list_after = []
        for layer in self.model.layers:
            weights_list_after.append(layer.get_weights())