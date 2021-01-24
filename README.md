# Code for Problem Description

Before running this code, clone this repository and use pip to install all required dependencies listed below:

* tensorflow
* tensorflow_datasets
* tensorflow_privacy
* numpy

To run this code, run

``` python
python transfer_model_fashion_mnist.py
```

This will load the pre-trained model in the model_0 directory and perform transfer learning for the Fashion-MNIST dataset.  More info about this dataset can be found [here](https://github.com/zalandoresearch/fashion-mnist).

If you wish to generate a fresh model saved in the model_0 directory, run

``` python
python base_model.py
```

This will train a model on the MNIST dataset and save it to the model_0 directory; however the point of providing the model saved in the model_0 directory in this git repository is to avoid having to train the base model over and over again in order to perform the transfer learning experiments.