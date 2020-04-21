import os
import pickle

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

root = "/mnt/data3/caojh/dataset/AstroData"

name = "correct"
with open(os.path.join(os.path.join(root, "training"), "train_XY_" + name + ".pkl"), 'rb') as f:
    X_train, Y_train = pickle.load(f)
    print(X_train.shape)

parameters = {
    'max_depth': [7, 8, 9, 10],
    'num_leaves': [80, 100, 120, 140],
}

gbm = lgb.LGBMClassifier(objective='multiclass',
                         is_unbalance=True,
                         metric='multi_logloss',
                         max_depth=6,
                         num_leaves=80,
                         learning_rate=0.1,
                         feature_fraction=0.7,
                         min_child_samples=21,
                         min_child_weight=0.001,
                         bagging_fraction=1,
                         bagging_freq=2,
                         reg_alpha=0.001,
                         reg_lambda=8,
                         cat_smooth=0,
                         num_iterations=200,
                         )
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='f1_macro', verbose=2, cv=3, n_jobs=40)
gsearch.fit(X_train, Y_train)
print('参数的最佳取值:{0}'.format(gsearch.best_params_))
print('最佳模型得分:{0}'.format(gsearch.best_score_))
print(gsearch.cv_results_['mean_test_score'])
print(gsearch.cv_results_['params'])
