import argparse
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Dropout, Conv2D, AveragePooling2D, Flatten, concatenate, \
    BatchNormalization, Activation
from keras.models import Model
from keras.utils import np_utils
from scipy.special import softmax
from sklearn.metrics import f1_score, confusion_matrix

from preprocess.integrate import build_features
from preprocess.savgol import savgol_smooth
from preprocess.standardize import standa
import pandas as pd


def ConvNet(input_shape=(2600, 1, 1), classes=3) -> Model:
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


# In[0]
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


# In[1]

def get_valdata_gbm_by_build(root):
    print("Building val data...")
    with open(os.path.join(os.path.join(root, "validation"), "val_sets_v1.pkl"), 'rb') as f:
        df_test_fea = pickle.load(f)
        print(df_test_fea)
    with open(os.path.join(os.path.join(root, "validation"), "val_labels_v1.pkl"), 'rb') as f:
        df_test_label = pickle.load(f)
        print(df_test_label)
    with open(os.path.join(os.path.join(root, "model"), "model_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    X_test = df_test_fea.iloc[:, 0:2600].values
    Y_test = df_test_label.iloc[:, 1:2].values.reshape(-1)
    X_test = build_features(X_test, pca)
    with open(os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl"), 'wb') as f:
        pickle.dump((X_test, Y_test), f, protocol=4)
        print("File saved in", os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl"))
    return X_test, Y_test


def get_valdata_gbm_by_load(root):
    print("Reading {}".format(os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl")))
    with open(os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl"), 'rb') as f:
        data_test = pickle.load(f)
    return data_test


def load_valdata_cnn(root):
    with open(os.path.join(os.path.join(root, 'validation'), 'val_sets_v1.pkl'), 'rb') as f:
        df_test_fea = pickle.load(f)
        print(df_test_fea)
    with open(os.path.join(os.path.join(root, 'validation'), 'val_labels_v1.pkl'), 'rb') as f:
        df_test_label = pickle.load(f)
        print(df_test_label)

    X_test = df_test_fea.iloc[:, 0:2600].values
    Y_test = df_test_label.iloc[:, 1:2].values.reshape(-1)

    X_test = standa(X_test, method='unit')
    X_test = savgol_smooth(X_test)

    X_test = X_test.reshape(-1, X_test.shape[1], 1, 1)
    Y_test = np_utils.to_categorical(Y_test, 3)
    return X_test, Y_test


def get_testdata_gbm_by_build(root):
    print("Building test data...")
    with open(os.path.join(os.path.join(root, "test"), "test_sets.pkl"), 'rb') as f:
        df_test = pickle.load(f)
        print(df_test)
    with open(os.path.join(os.path.join(root, "model"), "model_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    X_test = df_test.iloc[:, 0:2600].values
    X_test = build_features(X_test, pca)
    with open(os.path.join(os.path.join(root, "test"), "test_X" + ".pkl"), 'wb') as f:
        pickle.dump(X_test, f, protocol=4)
        print("X_test saved in", os.path.join(os.path.join(root, "test"), "test_X" + ".pkl"))
    return X_test


def get_testdata_gbm_by_load(root):
    print("Reading {}".format(os.path.join(os.path.join(root, "test"), "test_X" + ".pkl")))
    with open(os.path.join(os.path.join(root, "test"), "test_X" + ".pkl"), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(os.path.join(root, "test"), "test_id" + ".pkl"), 'rb') as f:
        id_test = pickle.load(f)
    return X_test, id_test


def load_testdata_cnn(root):
    with open(os.path.join(os.path.join(root, 'test'), 'test_sets.pkl'), 'rb') as f:
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
    parser.add_argument("--load-val", action="store_true",
                        help="Load val data directly")
    parser.add_argument("--load-test", action="store_true",
                        help="Load test data directly")
    args = parser.parse_args()

    root = "/mnt/data3/caojh/dataset/AstroData"

    # In[1]
    cnn_1 = ConvNet()
    cnn_2 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_3 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_4 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_5 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_6 = ConvNet_BN(fc_layers=[1024, 512, 256])
    cnn_7 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)
    cnn_8 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)
    cnn_9 = ConvNet_BN(fc_layers=[1024, 512, 256], width=6)
    # cnn.compile(loss='categorical_crossentropy',
    #               optimizer='Adam(1e-3)',
    #               metrics=['accuracy'])

    cnn_1.load_weights(os.path.join(root, "model/checkpoint_correct_03-30-00-11.h5"))
    cnn_2.load_weights(os.path.join(root, "model/checkpoint_correct_03-30-11-30-32.h5"))
    cnn_3.load_weights(os.path.join(root, "model/checkpoint_correct_03-30-15-30-41.h5"))
    cnn_4.load_weights(os.path.join(root, "model/checkpoint_aug_correct_3_03-31-15-52-45.h5"))
    cnn_5.load_weights(os.path.join(root, "model/checkpoint_aug_correct_3_03-31-23-32-45.h5"))
    cnn_6.load_weights(os.path.join(root, "model/checkpoint_aug_correct_6_03-31-15-22-17.h5"))
    cnn_7.load_weights(os.path.join(root, "model/checkpoint_aug_correct_3_04-01-07-50-30.h5"))
    cnn_8.load_weights(os.path.join(root, "model/checkpoint_aug_correct_6_04-01-15-01-31.h5"))
    cnn_9.load_weights(os.path.join(root, "model/checkpoint_correct_04-01-15-03-10.h5"))

    # gbm_1 = lgb.Booster(model_file=os.path.join(root, "model/gbm.model"))
    # gbm_2 = lgb.Booster(model_file=os.path.join(root, "model/gbm_aug_correct_6.model"))

    # In[1]

    # if not args.load_val:
    #     data_val = get_valdata_gbm_by_build(root)
    # else:
    #     data_val = get_valdata_gbm_by_load(root)
    #
    # X_val, Y_val = load_valdata_cnn(root)

    # In[1]
    # val_Y_pred_list = []
    # val_Y_pred_list.append(softmax(cnn_1.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_3.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_4.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_5.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_7.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_8.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(cnn_9.predict(X_val), axis=1))
    # val_Y_pred_list.append(softmax(gbm_2.predict(data_val[0], num_iteration=gbm_2.best_iteration), axis=1))

    # In[1]
    # factor_list = [[1, 1, 1],
    #                [5, 1, 1],
    #                [1, 5, 1],
    #                [4, 3, 3],
    #                [1, 1, 4],
    #                [1, 4, 1],
    #                [3, 1, 5],
    #                [0, 0, 0]]
    # factor_list = [[1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1],
    #                [1, 1, 1],
    #                [0, 0, 0]]

    # In[2]
    # from copy import deepcopy
    #
    # val_Y_pred_list_original = deepcopy(val_Y_pred_list)
    # for k in range(len(val_Y_pred_list_original)):
    #     print('-' * 40 + str(k) + ','  + '-' * 40)
    #     val_Y_pred_list = deepcopy(val_Y_pred_list_original)
    #     val_Y_pred_list.pop(k)

    # weighted_val_Y_pred_list = []
    # for i, val_Y_pred in enumerate(val_Y_pred_list):
    #     weighted_val_Y_pred_list.append(val_Y_pred * factor_list[i])
    # val_Y_pred_averaging = np.average(np.array(val_Y_pred_list), axis=0)
    # val_Y_pred_voting_list = []
    # for i, val_Y_pred in enumerate(val_Y_pred_list):
    #     val_Y_pred_voting_list.append(pd.get_dummies(val_Y_pred.argmax(1)).values)
    #     # val_Y_pred_voting_list.append(pd.get_dummies(val_Y_pred.argmax(1)).values * np.array(factor_list[i]))
    # val_Y_pred_voting = np.array(val_Y_pred_voting_list).sum(axis=0)
    #
    #
    # for i, val_Y_pred in enumerate(val_Y_pred_list):
    #     print("-" * 20 + "Classifier {} Validation score:\nmacro f1: {}".format(
    #         i, f1_score(data_val[1], val_Y_pred.argmax(axis=1), average="macro")))
    #     print(confusion_matrix(data_val[1], val_Y_pred.argmax(axis=1)))
    #
    # print("-" * 20 + "Averaging Validation score:\nmacro f1: {}".format(
    #     f1_score(data_val[1], val_Y_pred_averaging.argmax(axis=1), average="macro")))
    # print(confusion_matrix(data_val[1], val_Y_pred_averaging.argmax(axis=1)))
    #
    # print("-" * 20 + "Voting Validation score:\nmacro f1: {}".format(
    #     f1_score(data_val[1], val_Y_pred_voting.argmax(axis=1), average="macro")))
    # print(confusion_matrix(data_val[1], val_Y_pred_voting.argmax(axis=1)))

    # In[3]
    # val_Y_pred_final = val_Y_pred_averaging.argmax(axis=1)
    # cnt = 0
    # for i in range(val_Y_pred_final.shape[0]):
    #     if val_Y_pred_final[i] != data_val[1][i]:
    #         cnt += 1
    #         print(cnt, i, data_val[1][i], val_Y_pred_final[i], val_Y_pred_averaging[i])
    #
    # # In[3]
    # val_Y_pred_sort_idxs = np.argsort(val_Y_pred_averaging.max(axis=1))
    # cnt_true = 0
    # cnt_false = 0
    #
    # idx_list = []
    # pred_0_list = []
    # pred_1_list = []
    # pred_2_list = []
    # pred_final_list = []
    # label_list = []
    # for idx in val_Y_pred_sort_idxs[0:200]:
    #     if val_Y_pred_final[idx] == 2:
    #         print(idx, val_Y_pred_averaging[idx], val_Y_pred_final[idx], data_val[1][idx])
    #         idx_list.append(idx)
    #         pred_0_list.append(val_Y_pred_averaging[idx][0])
    #         pred_1_list.append(val_Y_pred_averaging[idx][1])
    #         pred_2_list.append(val_Y_pred_averaging[idx][2])
    #         pred_final_list.append(val_Y_pred_final[idx])
    #         label_list.append(data_val[1][idx])
    #         if val_Y_pred_final[idx] == data_val[1][idx]:
    #             cnt_true += 1
    #         else:
    #             cnt_false += 1
    # print(cnt_true, cnt_false)
    # df_minoutput = pd.DataFrame({'index': idx_list,
    #                              'confidence_0': pred_0_list,
    #                              'confidence_1': pred_1_list,
    #                              'confidence_2': pred_2_list,
    #                              'predict': pred_final_list,
    #                              'label': label_list})
    # df_minoutput.to_csv("minconfidence_2.csv", index=None)

    # In[4]
    # if not args.load_test:
    #     X_test_gbm = get_testdata_gbm_by_build(root)
    # else:
    #     X_test_gbm = get_testdata_gbm_by_load(root)

    X_test_cnn, id_test = load_testdata_cnn(root)

    # 注意,保持验证fusion和测试fusion一致!
    print("注意,保持验证fusion和测试fusion一致!")


    test_Y_pred_list = []
    test_Y_pred_list.append(softmax(cnn_1.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_3.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_4.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_5.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_7.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_8.predict(X_test_cnn), axis=1))
    test_Y_pred_list.append(softmax(cnn_9.predict(X_test_cnn), axis=1))

    # test_Y_pred_list = []
    # test_Y_pred_list.append(softmax(cnn_1.predict(X_test_cnn), axis=1))
    # test_Y_pred_list.append(softmax(cnn_3.predict(X_test_cnn), axis=1))
    # test_Y_pred_list.append(softmax(cnn_4.predict(X_test_cnn), axis=1))
    # test_Y_pred_list.append(softmax(cnn_5.predict(X_test_cnn), axis=1))
    # test_Y_pred_list.append(softmax(cnn_7.predict(X_test_cnn), axis=1))
    # test_Y_pred_list.append(softmax(gbm_2.predict(X_test_gbm, num_iteration=gbm_2.best_iteration), axis=1))
    # In[5]
    test_Y_pred_averaging = np.average(np.array(test_Y_pred_list), axis=0)
    # test_Y_pred_voting_list = []
    # for test_Y_pred in test_Y_pred_list:
    #     test_Y_pred_voting_list.append(pd.get_dummies(test_Y_pred.argmax(1)).values)
    # test_Y_pred_voting = np.array(test_Y_pred_voting_list).sum(axis=0)

    # In[5]
    df_test_Y_pred = pd.DataFrame({'id': id_test, 'label': test_Y_pred_averaging.argmax(axis=1)})
    num_to_label = {0: "star", 1: "galaxy", 2: "qso"}
    df_test_Y_pred['label'] = df_test_Y_pred['label'].map(num_to_label)

    df_test_Y_pred.to_csv("result_check.csv", index=None)
    print("输出完成, 注意,保持验证fusion和测试fusion一致!")

    # In[6]
    # test_Y_pred_final = test_Y_pred_averaging.argmax(axis=1)
    #
    # test_Y_pred_sort_idxs = np.argsort(test_Y_pred_averaging.max(axis=1))
    #
    # idx_list = []
    # pred_0_list = []
    # pred_1_list = []
    # pred_2_list = []
    # pred_final_list = []
    # for idx in test_Y_pred_sort_idxs[0:200]:
    #     if test_Y_pred_final[idx] == 0:
    #         print(idx, test_Y_pred_averaging[idx], test_Y_pred_final[idx])
    #         idx_list.append(idx)
    #         pred_0_list.append(test_Y_pred_averaging[idx][0])
    #         pred_1_list.append(test_Y_pred_averaging[idx][1])
    #         pred_2_list.append(test_Y_pred_averaging[idx][2])
    #         pred_final_list.append(test_Y_pred_final[idx])
    # df_minoutput = pd.DataFrame({'index': idx_list,
    #                              'confidence_0': pred_0_list,
    #                              'confidence_1': pred_1_list,
    #                              'confidence_2': pred_2_list,
    #                              'predict': pred_final_list})
    # df_minoutput.to_csv("test_minconfidence_0.csv", index=None)
