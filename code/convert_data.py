# In[0]
import time

import pandas as pd
import os
import pickle

root = "/mnt/data3/caojh/dataset/AstroData/training"

# csv转pkl
suffix = "correct"
answer_to_num = {"star": 0, "galaxy": 1, "qso": 2}
#for suffix in ["small", "medium", "correct"]:
name = "trains_sets_" + suffix
print(name)
start_time = time.time()
df = pd.read_csv(os.path.join(root, name + ".csv"))
print("read {}: {} sec".format(name + ".csv", time.time() - start_time))
print(df.shape)
df['answer'] = df['answer'].map(answer_to_num)
df = df.drop("id", axis=1)
with open(os.path.join(root, name + ".pkl"), 'wb') as f:
    pickle.dump(df, f, protocol=4)
print("file saved in " + os.path.join(root, name + ".pkl"))
start_time = time.time()
with open(os.path.join(root, name + ".pkl"), 'rb') as f:
    df1 = pickle.load(f)
print("read {}: {} sec".format(name + ".pkl", time.time() - start_time))
print(df1.shape)

# # answer列字符串转数字
# answer_to_num = {"star": 0, "galaxy": 1, "qso": 2}
# for suffix in ["small", "medium", "correct"]:
#     name = "trains_sets_" + suffix
#     print(name)
#     with open(os.path.join(root, name + ".pkl"), 'rb') as f:
#         df = pickle.load(f)
#         df['answer'] = df['answer'].map(answer_to_num)
#         with open(os.path.join(root, name + ".pkl"), 'wb') as f:
#             pickle.dump(df, f, protocol=4)
#
# # 删除id列
# # for suffix in ["small", "medium", "correct"]:
#     name = "trains_sets_" + suffix
#     print(name)
#     with open(os.path.join(root, name + ".pkl"), 'rb') as f:
#         df = pickle.load(f)
#         df = df.drop("id", axis=1)
#         print(df)
#         with open(os.path.join(root, name + ".pkl"), 'wb') as f:
#             pickle.dump(df, f, protocol=4)
# In[1]
# root = "/mnt/data3/caojh/dataset/AstroData/validation"
# name = "val_labels_v1"
# start_time = time.time()
# df = pd.read_csv(os.path.join(root, name + ".csv"))
# print("read {}: {} sec".format(name + ".csv", time.time() - start_time))
# print(df.shape)
# # In[2]
# label_to_num = {"star": 0, "galaxy": 1, "qso": 2}
# df['label'] = df['label'].map(label_to_num)
# with open(os.path.join(root, name + ".pkl"), 'wb') as f:
#     pickle.dump(df, f, protocol=4)
# # In[3]
# start_time = time.time()
# with open(os.path.join(root, name + ".pkl"), 'rb') as f:
#     df1 = pickle.load(f)
# print("read {}: {} sec".format(name + ".pkl", time.time() - start_time))
# print(df1.shape)
