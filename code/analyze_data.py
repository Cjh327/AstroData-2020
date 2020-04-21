# In[0]

import os

import pandas as pd

root = "/mnt/data3/caojh/dataset/AstroData/training"

# In[1]
df = pd.read_csv(os.path.join(root, "trains_sets_small.csv"))

# In[2]
print(df)
print(df['answer'].value_counts())

df_star = df[df['answer'] == 'star']
df_galaxy = df[df['answer'] == 'galaxy']
df_qso = df[df['answer'] == 'qso']
# print(df_star)
# print(df_galaxy)
# print(df_qso)

# In[3]
import matplotlib.pyplot as plt

for df_plot, name in zip([df_star, df_galaxy, df_qso], ["star", "galaxy", "qso"]):
    plt.figure()
    for i in range(5):
        feature = df_plot.iloc[i:i + 1, 0:2600].to_numpy()
        feature = feature.reshape(-1)
        print(name, i, feature)
        assert feature.shape == (2600,)
        plt.plot(feature)
    plt.title(name)
    plt.savefig(os.path.join("fig", name + ".png"))
    plt.show()
    plt.close()


# In[5]
# from scipy.signal import savgol_filter
#
# feature = df_star.iloc[0:1, 0:2600].to_numpy()
# feature_smooth = savgol_filter(feature, window_length=501, polyorder=3)
# feature_smooth = feature_smooth.reshape(-1)
#
# # In[6]
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(feature_smooth)
# plt.title("star_0_smooth")
# plt.savefig(os.path.join("fig", "star_0_smooth" + ".png"))
# plt.show()
# plt.close()

# In[7]
# df_medium = pd.read_csv(os.path.join(root, "trains_sets_medium.csv"))
# data = df_medium.iloc[:, 0:2600].to_numpy()
# print(data.shape)
#
# # In[8]
# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=750, copy=True)
# new_data = pca.fit_transform(data)
# print(pca.explained_variance_ratio_.sum())

# In[9]
