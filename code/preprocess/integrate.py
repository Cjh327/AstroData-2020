import sys

sys.path.append("..")

import numpy as np
import copy
from scipy.signal import savgol_filter

from preprocess.interval import interval_stat
from preprocess.standardize import standa


def preprocess_x0(features, pca):
    """
    Preprocess X_0: PCA
    :param features: raw features
    :param pca: pca model
    :return: X_0 features
    """
    print("Preprocess X0")
    x_raw = copy.deepcopy(features)
    x0 = pca.transform(x_raw)
    print("X0 shape:", x0.shape)
    return x0


def preprocess_x1(features):
    """
    Preprocess X_1: savgol smoothing + standardize + interval
    :param features: raw features
    :return: X_1 features
    """
    print("Preprocess X1")
    x_raw = copy.deepcopy(features)
    x1 = standa(x_raw, method="unit")
    # x1 = remove_abnormal(x1)

    x1 = savgol_filter(x1, window_length=7, polyorder=3)
    x1 = interval_stat(x1, 50)
    print("X1 shape:", x1.shape)
    return x1


def cat_features(features_list):
    return np.concatenate(features_list, axis=1)


def build_features(features, pca):
    x0 = preprocess_x0(features, pca)
    x1 = preprocess_x1(features)
    x_build = cat_features([x0, x1])
    print("Build feature shape:", x_build.shape)
    return x_build
