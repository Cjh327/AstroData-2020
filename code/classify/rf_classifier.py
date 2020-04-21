# In[0]
import os
import pickle
import sys
import time

sys.path.append("..")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from preprocess.savgol import savgol_smooth
from preprocess.standardize import standa


def preprocess_data(data, pca):
    print("start preprocessing")
    features, labels = data
    start_time = time.time()
    features = savgol_smooth(features)
    features = standa(features, method="unit")
    if pca is not None:
        features = pca.transform(features)
    print("Preprocess time:", time.time() - start_time)
    return (features, labels)


def train(data):
    print("start training")
    features, labels = data
    clf = RandomForestClassifier(n_estimators=10)
    start_time = time.time()
    clf.fit(features, labels)
    print("Train time:", time.time() - start_time)
    y_hat = clf.predict(features)
    score = {"accuracy": accuracy_score(labels, y_hat),
             "f1": f1_score(labels, y_hat, average="macro")}
    print("Train score:", score)
    return clf


def test(data, clf):
    features, labels = data
    y_hat = clf.predict(features)
    score = {"accuracy": accuracy_score(labels, y_hat),
             "f1": f1_score(labels, y_hat, average="macro")}
    print("Test score:", score)
    print(classification_report(labels, y_hat))
    return y_hat


def write_result(filepath, y_hat, id_list):
    result = {"id": id_list,
              "label": y_hat}
    label_to_str = {0: "star", 1: "galaxy", 2: "qso"}
    df = pd.DataFrame(result)
    df['label'] = df['label'].map(label_to_str)
    print(df)
    df.to_csv(filepath, index=False)
    return df


# In[1]
if __name__ == "__main__":
    root = "/mnt/data3/caojh/dataset/AstroData"

    with open(os.path.join(os.path.join(root, "training"), "trains_sets_correct_pca.pkl"), 'rb') as f:
        df_full = pickle.load(f)
    print(df_full)

    with open(os.path.join(os.path.join(root, "validation"), "val_sets_v1.pkl"), 'rb') as f:
        df_val_fea = pickle.load(f)
    print(df_val_fea)

    with open(os.path.join(os.path.join(root, "validation"), "val_labels_v1.pkl"), 'rb') as f:
        df_val_label = pickle.load(f)
    print(df_val_label)

    # In[2]
    # pca = None
    with open(os.path.join(os.path.join(root, "model"), "model_pca.pkl"), 'rb') as f:
        pca = pickle.load(f)

    # In[3]
    # split_num = int(4 * df_full.shape[0] / 5)
    # features_train = df_full.iloc[0:split_num, 0:2600].values
    # labels_train = df_full.iloc[0:split_num, 2600:2601].values.reshape(-1)
    # features_test = df_full.iloc[split_num:, 0:2600].values
    # labels_test = df_full.iloc[split_num:, 2600:2601].values.reshape(-1)

    features_train = df_full.iloc[:, 0:1500].values
    labels_train = df_full.iloc[:, 1500:1501].values.reshape(-1)
    data_train = (features_train, labels_train)

    # In[2]
    features_test = df_val_fea.iloc[:, 0:2600].values
    labels_test = df_val_label.iloc[:, 1:2].values.reshape(-1)

    # data_train = preprocess_data((features_train, labels_train), pca)
    data_test = preprocess_data((features_test, labels_test), pca)

    print(data_train[0].shape, data_test[0].shape)

    # In[3]
    clf = train(data_train)

    # In[4]
    y_hat = test(data_test, clf)
    # In[5]
    # with open(os.path.join(root, "model_rf_891567.pkl"), 'wb') as f:
    #     pickle.dump(clf, f, protocol=4)

    # In[6]
    # id_list = df_val_fea["id"].values
    # df_result = write_result(os.path.join(os.path.join(root, "result"), "result.csv"), y_hat, id_list)
