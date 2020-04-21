# In[0]

import numpy as np

from preprocess.interval import load_df
from preprocess.standardize import standa


def remove_abnormal(x_train: np.ndarray) -> np.ndarray:
    rows, columns = x_train.shape

    mean, sigma = x_train.mean(axis=0), x_train.std(axis=0)
    cond1 = (mean - 6 * sigma < x_train)
    cond2 = (x_train < mean + 6 * sigma)  # type: np.ndarray
    for i in range(rows):
        if cond1[i].all() and cond2[i].all():
            continue
        for j in range(1, columns - 1):
            if not (cond1[i][j] and cond2[i][j]):
                x_train[i][j] = (x_train[i][j - 1] + x_train[i][j + 1])/2
    return x_train


if __name__ == '__main__':
    root = "/mnt/data3/caojh/dataset/AstroData/"
    train_file = 'trains_sets_correct.pkl'
    test_file = 'val_labels_v1.pkl'
    X_train, Y_train = load_df(root + 'training', train_file)  # type: (np.ndarray, np.ndarray)
    X_train = standa(X_train)
    X_train = remove_abnormal(X_train)

