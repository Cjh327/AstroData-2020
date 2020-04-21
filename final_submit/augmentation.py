# In[0]
import copy
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append("..")


def data_augmentation(df, frac_list, aug_num):
    df_aug_all = copy.deepcopy(df)
    for i, frac in enumerate(frac_list):
        for j in range(aug_num):
            print("-" * 40 + str(i) + " " + str(j) + "-" * 40)
            df_aug = df[df["answer"] == i]
            df_aug = df_aug.sample(frac=frac)
            X = df_aug.iloc[:, 0:2600].values
            L = np.random.normal(0, 1, (X.shape[0], 2600))
            s = X - np.mean(X, axis=1).reshape(-1, 1)
            k = 0.2
            X += L * s * k
            df_aug.iloc[:, 0:2600] = X
            print(df_aug.shape)
            df_aug_all = pd.concat([df_aug_all, df_aug]).reset_index(drop=True)
    print(df["answer"].value_counts())
    print(df_aug_all["answer"].value_counts())
    return df_aug_all


if __name__ == "__main__":
    root = "/mnt/data3/caojh/dataset/AstroData"
    name = "correct"

    with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + ".pkl"), 'rb') as f:
        df = pickle.load(f)
    print(df["answer"].value_counts())
    aug_num = 3
    df_aug = data_augmentation(df, [0.001, 1, 0.5], aug_num=aug_num)

    with open(os.path.join(os.path.join(root, "training"),
                           "trains_sets_aug_{}_{}.pkl".format(name, aug_num)), 'wb') as f:
        pickle.dump(df_aug, f, protocol=4)
