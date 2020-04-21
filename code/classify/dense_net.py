# In[0]
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv2D, AveragePooling2D, Flatten, concatenate, \
    BatchNormalization, Activation

from keras.initializers import glorot_uniform
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

import os
import pickle
import numpy as np
import pandas as pd
import datetime

from preprocess.savgol import savgol_smooth
from preprocess.standardize import standa

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.utils import class_weight


def ConvNet_DropOut(input_shape=(2600, 1, 1), classes=3) -> Model:
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
    for nodes in [1024, 512, 256]:
        X = Dense(nodes, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
        X = Dropout(0.8)(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed))(X)
    model = Model(inputs=X_input, outputs=X, name='ConvNet')
    return model


def ConvNet_BN(input_shape=(2600, 1, 1), classes=3) -> Model:
    seed = 710
    X_input = Input(input_shape)
    X_list = []
    for step in range(3, 19, 2):
        Xi = X_input
        for _ in range(6):
            '''
            Xi = Conv2D(32, (step, 1), strides=(1, 1),
                        padding='same', activation='relu', kernel_initializer=glorot_uniform(seed))(Xi)
            '''
            Xi = Conv2D(32, (step, 1), strides=(1, 1), use_bias=False,
                        padding='same', kernel_initializer=glorot_uniform(seed))(Xi)
            Xi = BatchNormalization()(Xi)
            Xi = Activation('relu')(Xi)
            Xi = AveragePooling2D((3, 1), strides=(3, 1))(Xi)
        X_list.append(Xi)
    X = concatenate(X_list, axis=3)
    X = Flatten()(X)
    '''
    X = Dense(1024, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.8)(X)
    X = Dense(512, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.8)(X)
    X = Dense(256, activation='relu', kernel_initializer=glorot_uniform(seed))(X)
    X = Dropout(0.8)(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed))(X)
    '''
    for nodes in fc_layers:
        X = Dense(nodes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
    X = Dense(classes, use_bias=False, kernel_initializer=glorot_uniform(seed))(X)
    X = BatchNormalization()(X)
    X = Activation('softmax')(X)
    model = Model(inputs=X_input, outputs=X, name='ConvNet')
    return model


def load_data(root, name):
    with open(os.path.join(os.path.join(root, 'training'), 'trains_sets_' + name + '.pkl'), 'rb') as f:
        df_train = pickle.load(f)
        print(df_train)
    with open(os.path.join(os.path.join(root, 'validation'), 'val_sets_v1.pkl'), 'rb') as f:
        df_test_fea = pickle.load(f)
        print(df_test_fea)
    with open(os.path.join(os.path.join(root, 'validation'), 'val_labels_v1.pkl'), 'rb') as f:
        df_test_label = pickle.load(f)
        print(df_test_label)

    X_train_all = df_train.iloc[:, 0:2600].values
    Y_train_all = df_train.iloc[:, 2600:2601].values.reshape(-1)
    X_test = df_test_fea.iloc[:, 0:2600].values
    Y_test = df_test_label.iloc[:, 1:2].values.reshape(-1)

    X_train_all = standa(X_train_all, method='unit')
    X_train_all = savgol_smooth(X_train_all)
    X_test = standa(X_test, method='unit')
    X_test = savgol_smooth(X_test)

    return X_train_all, Y_train_all, X_test, Y_test


def f1_metric(y_true, y_pred):
    y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
    score = 0
    for i in range(classes):
        p = K.cast(K.equal(y_true, i), dtype='float32')
        r = K.cast(K.equal(y_pred, i), dtype='float32')
        score += 2*K.sum(p*r)/(classes*K.sum(p + r + K.epsilon()))
    return score


def f1_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(3):
        p = K.constant([list(i)])*y_true
        r = K.constant([list(i)])*y_pred
        loss += 2*K.sum(p*r)/(classes*K.sum(p + r + K.epsilon()))
    return - K.log(loss + K.epsilon())


class Evaluate(Callback):

    def __init__(self):
        super().__init__()
        self.scores = []
        self.highest_score = 0

    def on_epoch_begin(self, epoch, logs=None):
        if len(self.scores) > 0:
            return
        self.on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        X_val, Y_val = self.validation_data[0], self.validation_data[1]
        Y_pred = model.predict(X_val)
        Y_val, Y_pred = np.argmax(Y_val, axis=1), np.argmax(Y_pred, axis=1)
        print("Test score:\nmacro f1: {}\n{}".format(f1_score(Y_val, Y_pred, average="macro"),
                                                     classification_report(Y_val, Y_pred)))
        print("Confusion matrix:\n{}".format(confusion_matrix(Y_val, Y_pred)))
        score = f1_score(Y_val, Y_pred, average='macro')
        self.scores.append(score)
        if score >= self.highest_score:
            self.highest_score = score
            model.save_weights(model_name)
        print('macro f1: {}, highest score: {}'.format(score, self.highest_score))


classes = 3

root = '/mnt/data3/caojh/dataset/AstroData'
name = 'correct'
time_stamp = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
model_name = 'model/checkpoint_{}_{}.h5'.format(name, time_stamp)

os.environ['CUDA_VISIBLE_DEVICES'] = '14'
batch_size = 128
fc_layers = [1024, 512, 256]


if __name__ == '__main__':
    model = ConvNet_BN()
    # model = multi_gpu_model(model, gpus=4)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3),
                  metrics=['accuracy'])

    print('start loading data')
    X_train, Y_train, X_test, Y_test = load_data(root, name)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1, 1))
    Y_train = np_utils.to_categorical(Y_train, 3)
    Y_test = np_utils.to_categorical(Y_test, 3)

    # X_train = np.concatenate((X_train, X_test), axis=0)
    # Y_train = np.concatenate((Y_train, Y_test), axis=0)
    print('finish loading data')

    evaluator = Evaluate()

    # model.load_weights(model_name)
    # print('loading from {}'.format(model_name))

    Y_train_one = np.argmax(Y_train, axis=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  epsilon=0.0001, patience=5, mode='auto')
    model.fit(x=X_train, y=Y_train, batch_size=batch_size, epochs=30, verbose=1,
              validation_data=(X_test, Y_test),
              callbacks=[evaluator, reduce_lr])
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    pd.DataFrame(Y_pred).to_csv('result.csv', index=False)

