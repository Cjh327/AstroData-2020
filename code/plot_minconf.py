import pandas as pd
import matplotlib.pyplot as plt
import pickle

from preprocess.savgol import savgol_smooth
from preprocess.standardize import standa

df_0 = pd.read_csv('test_minconfidence_0.csv')
df_1 = pd.read_csv('test_minconfidence_1.csv')
df_2 = pd.read_csv('test_minconfidence_2.csv')

with open('/mnt/data3/caojh/dataset/AstroData/test/test_sets.pkl', 'rb') as f:
    df_test_fea = pickle.load(f)
    print(df_test_fea)

# with open('X_test.pkl', 'rb') as f:
#     X_test = pickle.load(f)

X_test = df_test_fea.iloc[:, 0:2600].values

X_test = standa(X_test, method='unit')
print('standa')
X_test = savgol_smooth(X_test)
print('smooth')

with open("X_test.pkl", 'wb') as f:
    pickle.dump(X_test, f, protocol=4)
    print("X_test saved")

for i in range(len(df_0)):
    plt.figure()
    index = df_0['index'][i]
    feature = X_test[index].reshape(-1)
    print(index, feature)
    assert feature.shape == (2600,)
    plt.plot(range(2600), feature)
    # name = '0-{}-{}'.format(df_0['label'][i], df_0['index'][i])
    name = '0-{}'.format(df_0['index'][i])
    plt.title(name)
    plt.savefig("test_minconf0/{}.png".format(name))
    plt.show()
    plt.close()


for i in range(len(df_1)):
    plt.figure()
    index = df_1['index'][i]
    feature = X_test[index].reshape(-1)
    print(index, feature)
    assert feature.shape == (2600,)
    plt.plot(range(2600), feature)
    name = '1-{}'.format(df_1['index'][i])
    plt.title(name)
    plt.savefig("test_minconf1/{}.png".format(name))
    plt.show()
    plt.close()


for i in range(len(df_2)):
    plt.figure()
    index = df_2['index'][i]
    feature = X_test[index].reshape(-1)
    print(index, feature)
    assert feature.shape == (2600,)
    plt.plot(range(2600), feature)
    name = '2-{}'.format(df_2['index'][i])
    plt.title(name)
    plt.savefig("test_minconf2/{}.png".format(name))
    plt.show()
    plt.close()