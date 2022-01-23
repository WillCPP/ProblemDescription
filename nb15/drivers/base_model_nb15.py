import pandas as pd
import numpy as np
from tensorflow import keras

def base_model_nb15(save_model=False, model_folder=None, use_tf_privacy=False, noise_multiplier=None,  label_col='label'):
    # Model / data parameters
    if label_col == 'label':
        num_classes = 2
    elif label_col == 'attack_cat':
        num_classes = 10
    input_shape = (190, 1,)

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Read training data ==================================================
    df = pd.read_csv('nb15/data/UNSW_NB15_training-set.csv')
    # Prepare data
    df = df.drop(columns=df.columns[0], axis=1)
    df.loc[df['service'] == '-', 'service'] = 'none'
    # One-hot encode catagorical columns
    df = pd.get_dummies(df, columns=['proto', 'service', 'state'], prefix=['proto', 'service', 'state'])
    # Get labels col, and drop them from df
    col_labels = df['label']
    col_attack_cat = df['attack_cat']
    df = df.drop(columns=['label', 'attack_cat'], axis=1)
    if label_col == 'label':
        df_label = col_labels
    elif label_col == 'attack_cat':
        df_label = col_attack_cat
    y_train = np.array(df_label)
    # convert class vector to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    # Normalize (mi-max normalization) df
    df = (df-df.min())/(df.max()-df.min())
    x_train = df.to_numpy()

    # Read testing data ==================================================
    df = pd.read_csv('nb15/data/UNSW_NB15_testing-set.csv')
    # Prepare data
    df = df.drop(columns=df.columns[0], axis=1)
    df.loc[df['service'] == '-', 'service'] = 'none'
    # One-hot encode catagorical columns
    df = pd.get_dummies(df, columns=['proto', 'service', 'state'], prefix=['proto', 'service', 'state'])
    # Get labels col, and drop them from df
    col_labels = df['label']
    col_attack_cat = df['attack_cat']
    df = df.drop(columns=['label', 'attack_cat'], axis=1)
    if label_col == 'label':
        df_label = col_labels
    elif label_col == 'attack_cat':
        df_label = col_attack_cat
    y_test = np.array(df_label)
    # convert class vector to binary class matrices
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Normalize (mi-max normalization) df
    df = (df-df.min())/(df.max()-df.min())
    x_test = df.to_numpy()