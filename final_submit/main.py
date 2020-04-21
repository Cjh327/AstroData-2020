import argparse
import os
import pickle

import numpy as np
import pandas as pd
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Dropout, Conv2D, AveragePooling2D, Flatten, concatenate, \
    BatchNormalization, Activation
from keras.models import Model
from scipy.special import softmax

from savgol import savgol_smooth
from standardize import standa


def ConvNet_Dropout(input_shape=(2600, 1, 1), classes=3) -> Model:
    seed = 710
    X_input = Input(input_shape)
    X_list = []
    for step in range(3, 19, 2):
        Xi = X_input
        for _ in range(3):
            Xi = Conv2D(32, (step, 1), strides=(1, 1),
                        padding='same', activation='relu', kernel_initializer=glorot_uniform(seed))(Xi)
            Xi = AveragePooling2D((3, 1), strides=(3, 1))(Xi)
        X_list.append(Xi)
    X = concatenate(X_list, axis=3)
    X = Flatten()(X)
    X = Dense(1024, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.6)(X)
    X = Dense(512, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.6)(X)
    X = Dense(256, activation='tanh', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.6)(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed))(X)
    model = Model(inputs=X_input, outputs=X, name='ConvNet')
    return model


def ConvNet_BN(input_shape=(2600, 1, 1), classes=3, fc_layers=None, width=3) -> Model:
    seed = 710
    X_input = Input(input_shape)
    X_list = []
    for step in range(3, 19, 2):
        Xi = X_input
        for _ in range(width):
            Xi = Conv2D(32, (step, 1), strides=(1, 1), use_bias=False,
                        padding='same', kernel_initializer=glorot_uniform(seed))(Xi)
            Xi = BatchNormalization()(Xi)
            Xi = Activation('relu')(Xi)
            Xi = AveragePooling2D((3, 1), strides=(3, 1))(Xi)
        X_list.append(Xi)
    X = concatenate(X_list, axis=3)
    X = Flatten()(X)
    for nodes in fc_layers:
        X = Dense(nodes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
    X = Dense(classes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
    X = BatchNormalization()(X)
    X = Activation('softmax')(X)
    model = Model(inputs=X_input, outputs=X, name='ConvNet')
    return model


def load_testdata(filepath):
    print('Reading data from {} ...'.format(filepath))
    with open(filepath, 'rb') as f:
        df_test = pickle.load(f)
        print(df_test)

    X_test = df_test.iloc[:, 0:2600].values
    id_test = df_test.iloc[:, -1].values.reshape(-1)

    X_test = standa(X_test, method='unit')
    X_test = savgol_smooth(X_test)

    X_test = X_test.reshape(-1, X_test.shape[1], 1, 1)
    return X_test, id_test


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "13"
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="test data path")
    args = parser.parse_args()

    X_test, id_test = load_testdata(args.filepath)

    cnn_1 = ConvNet_Dropout()
    cnn_3 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_4 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_5 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_7 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)
    cnn_8 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)
    cnn_9 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)

    cnn_1.load_weights("model/checkpoint_1.h5")
    cnn_3.load_weights("model/checkpoint_3.h5")
    cnn_4.load_weights("model/checkpoint_4.h5")
    cnn_5.load_weights("model/checkpoint_5.h5")
    cnn_7.load_weights("model/checkpoint_7.h5")
    cnn_8.load_weights("model/checkpoint_8.h5")
    cnn_9.load_weights("model/checkpoint_9.h5")

    test_Y_pred_list = []
    test_Y_pred_list.append(softmax(cnn_1.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_3.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_4.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_5.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_7.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_8.predict(X_test), axis=1))
    test_Y_pred_list.append(softmax(cnn_9.predict(X_test), axis=1))

    test_Y_pred_averaging = np.average(np.array(test_Y_pred_list), axis=0)

    df_test_Y_pred = pd.DataFrame({'id': id_test, 'label': test_Y_pred_averaging.argmax(axis=1)})
    num_to_label = {0: "star", 1: "galaxy", 2: "qso"}
    df_test_Y_pred['label'] = df_test_Y_pred['label'].map(num_to_label)

    df_test_Y_pred.to_csv("result.csv", index=None)
