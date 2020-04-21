# In[0]

from classify import lgb_classifier
from preprocess.interval import *
from sklearn.model_selection import train_test_split

# In[1]


if __name__ == '__main__':
    root = "/mnt/data3/caojh/dataset/AstroData/"
    train_file = 'trains_sets_correct.pkl'

    X_train, Y_train = load_df(root + 'training', train_file)

    X_train = standa(X_train)
    X_train = savgol_filter(X_train, window_length=7, polyorder=3)
    X_train = interval_stat(X_train, 50)

    X_train, Y_train = (X_train[Y_train > 0], Y_train[Y_train > 0])
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)
    data_train, data_valid = (X_train, Y_train), (X_valid, Y_valid)

    fea_file = root + 'validation/val_sets_v1.pkl'
    label_file = root + 'validation/val_labels_v1.pkl'
    X_test, Y_test = load_validation(fea_file, label_file)

    X_test = standa(X_test)
    X_test = savgol_filter(X_test, window_length=7, polyorder=3)
    X_test = interval_stat(X_test, 50)

    data_test = (X_test[Y_test > 0], Y_test[Y_test > 0])
    print("Datasets shape:", data_train[0].shape, data_valid[0].shape, data_test[0].shape)

# In[2]
    # Set parameters
    lgb_classifier.print_log("Setting parameters...")
    other_params = {"boost_round": 1000,
                    "early_stop_rounds": 50
                    }
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',  # multiclassova
        # 'is_unbalance': True,     # works with objective multiclassova
        'num_class': 3,
        'num_iterations': 10000,
        'learning_rate': 0.1,
        'num_leaves': 100,
        'device_type': 'cpu',
        'metric': 'multi_logloss',
        'num_threads': 40,
        'max_depth': 8,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    print(params)
    print(other_params)

    lgb_classifier.print_log("Training model...")
    gbm = lgb_classifier.train(data_train, data_valid, params, other_params)

    # Test model
    lgb_classifier.print_log("Testing model ...")
    lgb_classifier.test(data_test, gbm)


