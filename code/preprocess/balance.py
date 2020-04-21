# In[0]
import os
import pickle
import time
from collections import Counter

import pandas as pd
from imblearn.combine import SMOTEENN


def balance_train_data(data):
    print("Start balancing...")
    features, labels = data

    start_time = time.time()
    smote_enn = SMOTEENN(random_state=42)
    features, labels = smote_enn.fit_sample(features, labels)
    print("Balanced dataset:", sorted(Counter(labels).items()))
    print("Balancing time:", time.time() - start_time)
    return (features, labels)


# In[1]

root = "/mnt/data3/caojh/dataset/AstroData"

for name in ["small", "medium", "correct"]:
    with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + ".pkl"), 'rb') as f:
        df = pickle.load(f)
    print(df)

    # df_tiny = df.sample(frac=0.1)
    # print(df_tiny["answer"].value_counts())

    features = df.iloc[:, 0:2600].values
    labels = df.iloc[:, 2600:2601].values.reshape(-1)

    (features, labels) = balance_train_data((features, labels))

    df_balanced = pd.DataFrame(features)
    df_balanced["answer"] = labels

    with open(os.path.join(os.path.join(root, "training"), "trains_sets_" + name + "_balanced.pkl"), 'wb') as f:
        pickle.dump(df_balanced, f, protocol=4)
        print("File saved in {}".format(
            os.path.join(os.path.join(root, "training"), "trains_sets_" + name + "_balanced.pkl")))
