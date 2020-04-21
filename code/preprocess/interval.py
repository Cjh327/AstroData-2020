# In[0]
import pickle
import os
import pandas as pd
import numpy as np

from scipy.signal import savgol_filter

from preprocess.standardize import standa
from classify.rf_classifier import train, test


wave_length = 2600


def load_df(root: str, fname: str) -> (np.ndarray, np.ndarray):
    with open(os.path.join(root, fname), 'rb') as f:
        df = pickle.load(f)  # type: pd.DataFrame
    feature = df.iloc[:, 0:-1].values
    label = df.iloc[:, -1:].values.reshape(-1)
    return feature, label


def load_validation(val_file: str, label_file: str) -> (np.ndarray, np.ndarray):
    with open(val_file, 'rb') as f1, open(label_file, 'rb') as f2:
        df_val_fea = pickle.load(f1)
        df_val_label = pickle.load(f2)
    X_test = df_val_fea.iloc[:, 0:2600].values
    Y_test = df_val_label.iloc[:, 1:2].values.reshape(-1)
    return X_test, Y_test


def interval_stat(feature: np.ndarray, n: int) -> np.ndarray:
    """
    :param feature:
    :param n:
    :return:
        将波长2600光谱等分为n份并计算其统计特征，
        包括：均值，方差，峰度，偏度，
             最大值频率，最小值频率
    """
    from math import ceil
    rows, columns = feature.shape
    step = ceil(columns/n)
    means, theta = np.zeros((rows, n)), np.zeros((rows, n))
    skews, kurts = np.zeros((rows, n)), np.zeros((rows, n))
    argmaxs, argmins = np.zeros((rows, n)), np.zeros((rows, n))
    mins, maxs = np.zeros((rows, n)), np.zeros((rows, n))
    for i in range(n):
        stop = min((i + 1)*step, columns)
        block = pd.DataFrame(feature[:, i*step:stop])
        means[:, i], theta[:, i] = block.mean(1), block.var(1)
        skews[:, i], kurts[:, i] = block.skew(1), block.kurt(1)
        argmaxs[:, i], argmins[:, i] = block.idxmax(1), block.idxmin(1)
        maxs[:, i], mins[:, i] = block.max(1), block.min(1)
    return np.concatenate((means, theta, skews, kurts,
                           argmaxs, argmins, maxs, mins), 1)

# In[1]


if __name__ == '__main__':
    root = "/mnt/data3/caojh/dataset/AstroData/"
    train_file = 'trains_sets_correct.pkl'
    test_file = 'val_labels_v1.pkl'
    X_train, Y_train = load_df(root + 'training', train_file)
    X_train = standa(X_train)
    X_train = savgol_filter(X_train, window_length=7, polyorder=3)
    X_train = interval_stat(X_train, 50)

    fea_file = root + 'validation/val_sets_v1.pkl'
    label_file = root + 'validation/val_labels_v1.pkl'
    X_test, Y_test = load_validation(fea_file, label_file)
    X_test = standa(X_test)
    X_test = savgol_filter(X_test, window_length=7, polyorder=3)
    X_test = interval_stat(X_test, 50)

    # In[1]
    clf = train((X_train, Y_train))
    test((X_test, Y_test), clf)
