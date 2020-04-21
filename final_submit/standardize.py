import numpy as np
import os
import pickle


def standa(X, method="unit"):
    """
    标准化光谱数据
    :param X: (m,n), m条数据，n个feature
    :param method: "unit": 单位化，"mean": 均值
    :return: 标准化后的数据 X_standa
    """
    print("standardize", method)
    if method == "unit":
        delta_X = np.sqrt(np.power(X, 2).sum(axis=1))
    elif method == "mean":
        delta_X = X.sum(axis=1) / X.shape[1]
    else:
        assert False
    X_standa = X / delta_X.reshape(-1, 1)
    return X_standa
