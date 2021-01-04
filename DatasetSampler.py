import numpy as np
import random

class DatasetSampler:
    def sample(self, X, Y, sample_size_per_class):
        labels = np.unique(Y)
        groups = {}
        for x, y in zip(X, Y):
            for label in labels:
                if y == label:
                    if y not in groups.keys():
                        groups[y] = []
                    groups[y].append(x)
        samples = {}
        for y, group in groups.items():
            samples[y] = random.sample(group, sample_size_per_class)
        ret_x_shape = [0]
        ret_x_shape.extend(list(X.shape[1:]))
        ret_X = np.empty(ret_x_shape)
        ret_y = np.array([])
        for y, sample in samples.items():
            ret_X = np.vstack((ret_X, sample))
            ret_y = np.hstack((ret_y, [y for i in range(0, len(sample))]))
        p = np.random.permutation(len(ret_X))
        return ret_X[p], ret_y[p]


# from tensorflow import keras
# from tensorflow.keras.datasets import fashion_mnist
# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)
# # the data, split between train and test sets
# (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
# X = np.vstack((X_train, X_test))
# y = np.hstack((y_train, y_test))
# ds = DatasetSampler()
# sample_X, sample_y = ds.sample(X, y, 20)
# print(sample_y)