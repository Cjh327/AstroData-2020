# In[0]
import argparse
import datetime
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.append("..")

from preprocess.integrate import build_features


def print_log(info):
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 4 + "%s" % now_time)
    print(info + '...\n')


def get_train_XY_by_build(root, name):
    print("Building train data...")
    """
    不划分 train set
    """
    with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + ".pkl"), 'rb') as f:
        df_train = pickle.load(f)
    print(df_train["answer"].value_counts())
    with open(os.path.join(os.path.join(root, "model"), "model_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    X_train = df_train.iloc[:, 0:2600].values
    Y_train = df_train.iloc[:, 2600:2601].values.reshape(-1)
    X_train = build_features(X_train, pca)
    with open(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"), 'wb') as f:
        pickle.dump((X_train, Y_train), f, protocol=4)
        print("File saved in", os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"))
    return (X_train, Y_train)


# def get_train_valid_XY_by_build(root, name, valid_size):
#     with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + ".pkl"), 'rb') as f:
#         df_train = pickle.load(f)
#         print(df_train)
#     # with open(os.path.join(os.path.join(root, "validation"), "val_sets_v1.pkl"), 'rb') as f:
#     #     df_test_fea = pickle.load(f)
#     #     print(df_test_fea)
#     # with open(os.path.join(os.path.join(root, "validation"), "val_labels_v1.pkl"), 'rb') as f:
#     #     df_test_label = pickle.load(f)
#     #     print(df_test_label)
#     with open(os.path.join(os.path.join(root, "model"), "model_pca.pkl"), "rb") as f:
#         pca = pickle.load(f)
#
#     X_train_all = df_train.iloc[:, 0:2600].values
#     Y_train_all = df_train.iloc[:, 2600:2601].values.reshape(-1)
#     # X_test = df_test_fea.iloc[:, 0:2600].values
#     # Y_test = df_test_label.iloc[:, 1:2].values.reshape(-1)
#
#     X_train_all = build_features(X_train_all, pca)
#     # X_test = build_features(X_test, pca)
#
#     with open(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"), 'wb') as f:
#         pickle.dump((X_train_all, Y_train_all), f, protocol=4)
#         print("File saved in", os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"))
#     # with open(os.path.join(os.path.join(root, "validation"), "test_XY_" + name + ".pkl"), 'wb') as f:
#     #     pickle.dump((X_test, Y_test), f, protocol=4)
#     #     print("File saved in", os.path.join(os.path.join(root, "validation"), "test_XY_" + name + ".pkl"))
#
#     X_train, X_val, Y_train, Y_val = train_test_split(X_train_all, Y_train_all,
#                                                       test_size=valid_size, random_state=42, shuffle=False)
#
#     return (X_train, Y_train), (X_val, Y_val)  # , (X_test, Y_test)


def get_test_XY_by_build(root, name):
    print("Building test data...")
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
    return (X_test, Y_test)


# def get_train_valid_XY_by_load(root, name, valid_size):
#     with open(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"), 'rb') as f:
#         data_train_all = pickle.load(f)
#     # with open(os.path.join(os.path.join(root, "validation"), "test_XY_" + name + ".pkl"), 'rb') as f:
#     #     data_test = pickle.load(f)
#     X_train, X_val, Y_train, Y_val = train_test_split(data_train_all[0], data_train_all[1],
#                                                       test_size=valid_size, random_state=42)
#
#     return (X_train, X_val), (Y_train, Y_val)  # , data_test


def get_train_XY_by_load(root, name):
    print("Reading {}".format(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl")))
    with open(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"), 'rb') as f:
        data_train = pickle.load(f)
    return data_train


def get_test_XY_by_load(root, name):
    print("Reading {}".format(os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl")))
    with open(os.path.join(os.path.join(root, "validation"), "test_XY" + ".pkl"), 'rb') as f:
        data_test = pickle.load(f)
    return data_test


# def plot_feature_importance(feature_importance):
#     assert feature_importance.shape[0] == 1150
#     n = 50
#     feature_names = ["pca", "means", "theta", "skews", "kurts", "argmaxs", "argmins", "maxs", "mins"]
#     importance_sum = feature_importance.sum()
#     importance_list = [feature_importance[0:750].sum() / importance_sum]
#     for i, fea_name in enumerate(feature_names[1:]):
#         importance = feature_importance[750 + n * i: 750 + n * (i + 1)].sum() / importance_sum
#         importance_list.append(importance)
#     print("Feature name list:", feature_names)
#     print("Feature importance list:", importance_list)
#     plt.figure()
#     plt.bar(range(len(importance_list)), importance_list, tick_label=feature_names)
#     plt.xticks(rotation=90)
#     for a, b in zip(range(len(importance_list)), importance_list):
#         plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=11)
#     plt.title("Feature importance")
#     plt.savefig("lgb_importance.png")
#     plt.show()

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = y_hat.reshape(len(np.unique(y_true)), -1).T.argmax(axis=1)
    return 'macro-f1', f1_score(y_true, y_hat, average="macro"), True


def train(data_train, data_valid, params, other_params):
    lgb_train = lgb.Dataset(data_train[0], label=data_train[1])
    lgb_valid = lgb.Dataset(data_valid[0], label=data_valid[1], reference=lgb_train)
    results = {}
    start_time = time.time()
    gbm = lgb.train(params,
                    lgb_train,
                    init_model=other_params["init_model"],
                    num_boost_round=other_params["boost_round"],
                    valid_sets=(lgb_valid, lgb_train),
                    valid_names=('validate', 'train'),
                    early_stopping_rounds=other_params["early_stop_rounds"],
                    feval=lgb_f1_score,
                    evals_result=results)
    print("Train time: {} sec".format(time.time() - start_time))
    Y_pred_train = gbm.predict(data_train[0], num_iteration=gbm.best_iteration).argmax(axis=1)
    print("Train score:\nmacro f1: {}\n{}".format(f1_score(data_train[1], Y_pred_train, average="macro"),
                                                  classification_report(data_train[1], Y_pred_train)))

    # plot_feature_importance(gbm.feature_importance())

    # Plot loss
    # plt.figure()
    # ax = lgb.plot_metric(results)
    # plt.savefig("lgb_metric.png")
    # plt.show()
    return gbm


def test(data_test, gbm):
    Y_pred_test = gbm.predict(data_test[0], num_iteration=gbm.best_iteration).argmax(axis=1)
    print("Test score:\nmacro f1: {}\n{}".format(f1_score(data_test[1], Y_pred_test, average="macro"),
                                                 classification_report(data_test[1], Y_pred_test)))
    print("Confusion matrix:\n{}".format(confusion_matrix(data_test[1], Y_pred_test)))
    return f1_score(data_test[1], Y_pred_test, average="macro")


def save_model(root, gbm, name):
    model_path = os.path.join(os.path.join(root, "model"), "gbm_{}.model".format(name))
    gbm.save_model(model_path, num_iteration=gbm.best_iteration)
    print("model path: %s" % model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-train", action="store_true",
                        help="Load train, validation data directly")
    parser.add_argument("--load-test", action="store_true",
                        help="Load test data directly")
    args = parser.parse_args()
    print(args.load_train, args.load_test)

    root = "/mnt/data3/caojh/dataset/AstroData"

    # Read data
    print_log("Reading data...")
    name = "aug_correct_3"
    if not args.load_train:
        data_train = get_train_XY_by_build(root, name)
    else:
        data_train = get_train_XY_by_load(root, name)
    if not args.load_test:
        data_test = get_test_XY_by_build(root, name)
    else:
        data_test = get_test_XY_by_load(root, name)

    print("Datasets shape:", data_train[0].shape, data_test[0].shape)

    # Set parameters
    print_log("Setting parameters...")
    other_params = {"boost_round": 1000,
                    "early_stop_rounds": 100,
                    "init_model": None  # os.path.join(os.path.join(root, "model"), "gbm.model")
                    }
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',  # multiclassova
        # 'is_unbalance': True,  # works with objective multiclassova
        # 'boost_from_average': False,  # works with objective multiclassova
        'num_class': 3,
        'num_iterations': 10000,
        'learning_rate': 0.1,
        'num_leaves': 100,
        'device_type': 'cpu',
        'metric': "None",
        'num_threads': 45,
        'max_depth': 8,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print(params)
    print(other_params)

    # Train model
    print_log("Training model...")
    gbm = train(data_train, data_test, params, other_params)

    # Test model
    print_log("Testing model ...")
    test(data_test, gbm)

    # # Save model
    print_log("Saving model ...")
    save_model(root, gbm, name)
    print_log("Task end...")
