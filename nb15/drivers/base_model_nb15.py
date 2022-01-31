import imp
import pandas as pd
import numpy as np
from tensorflow import keras
from DNN_nb15 import DNN
from parameters_nb15 import parameters
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from integer_encoder import C2I

def base_model_nb15(save_model=False, model_folder=None, use_tf_privacy=False, noise_multiplier=None,  label_col='label'):
    # Model / data parameters
    if label_col == 'label':
        num_classes = 2
    elif label_col == 'attack_cat':
        num_classes = 10
    # input_shape = (196, 1,)
    # input_shape = (190, 1,)
    # input_shape = (190,)
    input_shape = (42,)

    # Read training data ==================================================
    df = pd.read_csv('nb15/data/UNSW_NB15_training-set.csv')
    # Prepare data
    df = df.drop(columns=df.columns[0], axis=1)
    df.loc[df['service'] == '-', 'service'] = 'none'
    # Ordinal encoding categorical columns
    # df[['proto', 'service', 'state']] = df[['proto', 'service', 'state']].apply(lambda x: pd.factorize(x)[0])
    df['proto'] = C2I.encode_c2i(df['proto'], 'proto')
    df['service'] = C2I.encode_c2i(df['service'], 'service')
    df['state'] = C2I.encode_c2i(df['state'], 'state')
    # # One-hot encode catagorical columns
    # df = pd.get_dummies(df, columns=['proto', 'service', 'state'], prefix=['proto', 'service', 'state'])
    # # Add missing catagorical columns present in training data
    # df['proto_icmp'] = 0
    # df['proto_rtp'] = 0
    # df['state_ECO'] = 0
    # df['state_no'] = 0
    # df['state_PAR'] = 0
    # df['state_URN'] = 0
    # df['proto_icmp'] = df['proto_icmp'].astype('uint8')
    # df['proto_rtp'] = df['proto_rtp'].astype('uint8')
    # df['state_ECO'] = df['state_ECO'].astype('uint8')
    # df['state_no'] = df['state_no'].astype('uint8')
    # df['state_PAR'] = df['state_PAR'].astype('uint8')
    # df['state_URN'] = df['state_URN'].astype('uint8')
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
    # Reorder columns
    df = df.reindex(sorted(df.columns), axis=1)
    # Normalize (min-max normalization) df
    df = (df-df.min())/(df.max()-df.min())
    x_train = df.to_numpy()
    # Reshape data
    # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Read testing data ==================================================
    df = pd.read_csv('nb15/data/UNSW_NB15_testing-set.csv')
    # Prepare data
    df = df.drop(columns=df.columns[0], axis=1)
    df.loc[df['service'] == '-', 'service'] = 'none'
    # Ordinal encoding categorical columns
    # df[['proto', 'service', 'state']] = df[['proto', 'service', 'state']].apply(lambda x: pd.factorize(x)[0])
    df['proto'] = C2I.encode_c2i(df['proto'], 'proto')
    df['service'] = C2I.encode_c2i(df['service'], 'service')
    df['state'] = C2I.encode_c2i(df['state'], 'state')
    # # One-hot encode catagorical columns
    # df = pd.get_dummies(df, columns=['proto', 'service', 'state'], prefix=['proto', 'service', 'state'])
    # # Add missing catagorical columns present in training data
    # df['state_ACC'] = 0
    # df['state_CLO'] = 0
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
    # # Reorder columns
    # df = df.reindex(sorted(df.columns), axis=1)
    # Normalize (min-max normalization) df
    df = (df-df.min())/(df.max()-df.min())
    x_test = df.to_numpy()
    # Reshape data
    # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    if not use_tf_privacy:
        dnn = DNN(input_shape, num_classes, parameters['base_model'])
        dnn.createModel()
        dnn.train(x_train, y_train, x_test, y_test)
        if save_model: dnn.saveModel(model_folder + '/model_plain')
        score = dnn.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        return dnn.model

    if use_tf_privacy:
        dnn = DNN(input_shape, num_classes, parameters['base_model'], True, noise_multiplier)
        dnn.createModel()
        dnn.train(x_train, y_train, x_test, y_test)
        if save_model: dnn.saveModel(model_folder + '/model_private')
        score = dnn.model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        compute_dp_sgd_privacy.compute_dp_sgd_privacy(x_train.shape[0], \
                                                        parameters['base_model']['batch_size'], \
                                                        noise_multiplier, \
                                                        parameters['base_model']['epochs'],
                                                        1e-5)
        return dnn.model