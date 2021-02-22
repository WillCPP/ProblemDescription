import tensorflow_datasets as tfds
import numpy as np

emnist_train = tfds.load('emnist/mnist', split='train', batch_size=-1)
emnist_test = tfds.load('emnist/mnist', split='test', batch_size=-1)
emnist_train = tfds.as_numpy(emnist_train)
emnist_test = tfds.as_numpy(emnist_test)
(x_train, y_train) = emnist_train['image'], emnist_train['label']
(x_test, y_test) =  emnist_test['image'], emnist_test['label']

np.save('data/transfer_training_dataset/x_train.npy', x_train)
np.save('data/transfer_training_dataset/y_train.npy', y_train)
np.save('data/transfer_training_dataset/x_test.npy', x_test)
np.save('data/transfer_training_dataset/y_test.npy', y_test)
