import os
import pickle
import sys

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append("..")
from classify.lgb_classifier import print_log, train, test
from preprocess.integrate import preprocess_x0, preprocess_x1, cat_features
from preprocess.pca import train_pca

if __name__ == "__main__":
    root = "/mnt/data3/caojh/dataset/AstroData"
    name = "medium"

    print_log("Reading data...")

    with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + ".pkl"), 'rb') as f:
        df_train = pickle.load(f)
    with open(os.path.join(os.path.join(root, "validation"), "val_sets_v1.pkl"), 'rb') as f:
        df_test_fea = pickle.load(f)
    with open(os.path.join(os.path.join(root, "validation"), "val_labels_v1.pkl"), 'rb') as f:
        df_test_label = pickle.load(f)
        print(df_test_label)

    X_train_all = df_train.iloc[:, 0:2600].values
    Y_train_all = df_train.iloc[:, 2600:2601].values.reshape(-1)
    X_test = df_test_fea.iloc[:, 0:2600].values
    Y_test = df_test_label.iloc[:, 1:2].values.reshape(-1)

    print("get train X1")
    if not os.path.exists(os.path.join(os.path.join(root, "training"), "train_{}_X1.pkl".format(name))):
        X1_train_all = preprocess_x1(X_train_all)
        with open(os.path.join(os.path.join(root, "training"), "train_{}_X1.pkl".format(name)), 'wb') as f:
            pickle.dump(X1_train_all, f, protocol=4)
            print("train_X1 saved in", os.path.join(os.path.join(root, "training"), "train_{}_X1.pkl".format(name)))
    else:
        with open(os.path.join(os.path.join(root, "training"), "train_{}_X1.pkl".format(name)), 'rb') as f:
            X1_train_all = pickle.load(f)

    print("get test X1")
    if not os.path.exists(os.path.join(os.path.join(root, "validation"), "test_X1.pkl")):
        X1_test = preprocess_x1(X_test)
        with open(os.path.join(os.path.join(root, "validation"), "test_X1.pkl"), 'wb') as f:
            pickle.dump(X1_test, f, protocol=4)
            print("test_X1 saved in", os.path.join(os.path.join(root, "validation"), "test_X1.pkl"))
    else:
        with open(os.path.join(os.path.join(root, "validation"), "test_X1.pkl"), 'rb') as f:
            X1_test = pickle.load(f)

    # Set parameters
    print_log("Setting parameters...")
    other_params = {"boost_round": 1000,
                    "early_stop_rounds": 50,
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
        'verbose': -1
    }
    print(params)
    print(other_params)

    f1_list = []
    n_min, n_max = 100, 1000
    for n_components in range(n_min, n_max, 10):
        features = df_train.iloc[:, 0:2600].values
        model_path = os.path.join(os.path.join(root, "model"), "model_pca_tune.pkl")
        pca = train_pca(features, n_components, model_path)

        # Read data

        print("get train X0")
        X0_train_all = preprocess_x0(X_train_all, pca)

        print("get test X0")
        X0_test = preprocess_x0(X_test, pca)

        X_train_cat = cat_features([X0_train_all, X1_train_all])
        X_train, X_val, Y_train, Y_val = train_test_split(X_train_cat, Y_train_all,
                                                          test_size=0.2, random_state=42)
        X_test_cat = cat_features([X0_test, X1_test])

        data_train, data_valid, data_test = (X_train, Y_train), (X_val, Y_val), (X_test_cat, Y_test)

        print("Datasets shape:", data_train[0].shape, data_valid[0].shape, data_test[0].shape)

        # Train model
        print_log("Training model...")
        gbm = train(data_train, data_valid, params, other_params)

        # Test model
        print_log("Testing model ...")
        f1 = test(data_test, gbm)
        f1_list.append(f1)
        print_log("Task end...")

        plt.figure()
        plt.plot(range(n_min, n_components + 10, 10), f1_list)
        plt.xlim(n_min, n_components + 10)
        plt.title("Tune pca n_components")
        plt.savefig("tune_pca_{}_{}_{}.png".format(name, n_min, n_max))
        plt.show()
        plt.close()

        print(len(f1_list), f1_list)
    with open("f1_{}_{}_{}.pkl".format(name, n_min, n_max), 'wb') as f:
        pickle.dump(f1_list, f, protocol=4)
