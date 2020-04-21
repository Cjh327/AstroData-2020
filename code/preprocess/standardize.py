import pandas as pd
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    root = "/mnt/data3/caojh/dataset/AstroData"
    name = "trains_sets_small"
    with open(os.path.join(os.path.join(root, "training"), name + ".pkl"), 'rb') as f:
        df = pickle.load(f)

    df_star = df[df['answer'] == 0]
    df_galaxy = df[df['answer'] == 1]
    df_qso = df[df['answer'] == 2]

    for kind, df_plot in zip(["star", "galaxy", "qso"], [df_star, df_galaxy, df_qso]):
        feas = df_plot.iloc[0:5, 0:2600].values
        feas_standa = standa(feas, method="unit")
        plt.figure()
        for i in range(feas_standa.shape[0]):
            plt.plot(feas_standa[i].reshape(-1))
        plt.title(kind + "_standardized")
        plt.savefig(os.path.join("fig", kind + "_standardized" + ".png"))
        plt.show()
        plt.close()
