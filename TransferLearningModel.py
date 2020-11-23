from tensorflow import keras

class TransferLearningModel:
    def __init__(self):
        self.model = None
    
    def saveModel(self, path):
        self.model.save(path)

    def loadModel(self, path):
        self.model = keras.models.load_model(path)

    def fineTune(self, x_train, y_train, num_layers):
        # weights_list = []
        # for layer in self.model.layers:
        #     weights_list.append(layer.get_weights())

        for i in range(len(self.model.layers) - num_layers):
            self.model.layers[i].trainable = False
        
        self.model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
            loss=keras.losses.CategoricalCrossentropy(),#from_logits=True),
            metrics=[keras.metrics.CategoricalAccuracy()])
        
        self.model.fit(x_train, y_train, epochs=10, validation_split=0.1)

        weights_list_after = []
        for layer in self.model.layers:
            weights_list_after.append(layer.get_weights())

        # for i in range(len(weights_list)):
        #     print(f'Layer {i}')
        #     for wi, wj in zip(weights_list[i], weights_list_after[i]):
        #         print(f'i: {wi}')
        #         print(f'j: {wj}')