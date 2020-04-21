# In[0]
import time

import pandas as pd
import os
import pickle

root = "/mnt/data3/caojh/dataset/AstroData/test_sets"

# In[1]
df_list = []
start_time = time.time()
for i in range(10):
    df = pd.read_csv(os.path.join(root, "test_sets_{}.csv".format(i)))
    df.drop(df[df["FE0"] == "FE0"].index, inplace=True)
    print(i, df.shape)
    df_list.append(df)
print("time: {:.3} sec".format(time.time() - start_time))

# In[1.1]
df_test = pd.concat(df_list)
print(df_test.shape)

with open(os.path.join(root, "test_sets.pkl"), 'wb') as f:
    pickle.dump(df_test, f, protocol=4)
# df_test.to_csv(os.path.join(root, "test_sets.csv"), index=None)

# In[1.2]
# df = pd.read_csv(os.path.join(root, "test_sets.csv"))
# print(df.shape)
# with open(os.path.join(root, "test_sets.pkl"), 'wb') as f:
#     pickle.dump(df, f, protocol=4)

# In[2]
df_small = df.sample(frac=0.01)
print(df_small)
df_small.to_csv(os.path.join(root, "trains_sets_small.csv"), index=None)

# In[3]
df_medium = df.sample(frac=0.2)
print(df_medium)
df_medium.to_csv(os.path.join(root, "trains_sets_medium.csv"), index=None)

# In[4]
df1 = pd.read_csv(os.path.join(root, "trains_sets_small.csv"))
print(df1)
# print(df_small)

# In[5]
df2 = pd.read_csv(os.path.join(root, "trains_sets_medium.csv"))
print(df2)
# print(df_medium)

# In[6]
print(df_small['answer'].value_counts())

# In[7]
print(df_medium['answer'].value_counts())

# In[8]
print(df['answer'].value_counts())

# In[9]
print(df.shape)
df_correct = df[df['answer'] != 'answer']
print(df_correct.shape)

# In[10]
df_correct.to_csv(os.path.join(root, "trains_sets_correct.csv"), index=None)

# In[11]
root_val = "/mnt/data3/caojh/dataset/AstroData/validation/compressed"
df = pd.read_csv(os.path.join(root_val, "val_sets_v1.csv"))
print(df)

# In[12]
import pandas as pd
import os

root_val = "/mnt/data3/caojh/dataset/AstroData/validation"
df = pd.read_csv(os.path.join(root_val, "val_labels_v1.csv"))
print(df)

# In[13]
import pandas as pd
import os

df_0 = pd.read_csv("result_0.csv")
df_1 = pd.read_csv("result_1.csv")
df_2 = pd.read_csv("result_2.csv")
df = pd.read_csv("result.csv")
df_3 = pd.read_csv("result_3.csv")
same_cnt = (df_3['label'].values == df_2['label'].values).sum()
print(same_cnt, df_3.shape[0] - same_cnt, df_3.shape[0])
print(df['label'].value_counts())
print(df_3['label'].value_counts())
print(df_2['label'].value_counts())
print(df_1['label'].value_counts())
print(df_0['label'].value_counts())

root_val = "/mnt/data3/caojh/dataset/AstroData/validation"
df_val = pd.read_csv(os.path.join(root_val, "val_labels_v1.csv"))
print(df_val['label'].value_counts())

# In[1]
import pandas as pd
import numpy as np
import os

df = pd.read_csv("result.csv")
df_3 = pd.read_csv("result_3.csv")
print(np.where(df['label'].values != df_3['label'].values))
# print(df_3['label'][np.where(df['label'].values != df_3['label'].values)])
for idx in [32220, 117985, 18633, 25938, 31485, 45652, 58885, 59395, 69208, 84255, 100921, 109367, 137228, 166227,
            169236, 171441, 180893]:
    print(idx, df_3['label'].values[idx], df['label'].values[idx])
