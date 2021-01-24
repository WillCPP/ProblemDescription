from tensorflow import keras
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

class TransferLearningModel:
    def __init__(self):
        self.model = None
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = keras.models.load_model(path)

    def fineTune(self, x_train, y_train, params, private):
        for i in range(len(self.model.layers) - params['num_layers']):
            self.model.layers[i].trainable = False
        
        optimizer = None
        if private:
            optimizer = DPKerasAdamOptimizer(
            l2_norm_clip=params['l2_norm_clip'], 
            noise_multiplier=params['noise_multiplier'],
            learning_rate=params['learning_rate'])
        else:
            optimizer=keras.optimizers.Adam(0.001)
        
        self.model.compile(optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()])
        
        self.model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], validation_split=params['validation_split'])

        weights_list_after = []
        for layer in self.model.layers:
            weights_list_after.append(layer.get_weights())