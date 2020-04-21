import time

import pandas as pd
import pickle

df_list = []
start_time = time.time()
for i in range(10):
    df = pd.read_csv("test_sets_{}.csv".format(i))
    df.drop(df[df["FE0"] == "FE0"].index, inplace=True)
    print(i, df.shape)
    df_list.append(df)
print("time: {:.3} sec".format(time.time() - start_time))

df_test = pd.concat(df_list)
print(df_test.shape)

with open("test_sets.pkl", 'wb') as f:
    pickle.dump(df_test, f, protocol=4)
