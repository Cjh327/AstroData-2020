import os
import pickle
import sys

sys.path.append("..")

from sklearn.decomposition import PCA


def train_pca(features, n_components, model_path):
    pca = PCA(n_components=n_components, copy=True)
    features = pca.fit_transform(features)
    print("pca features:", features.shape)
    print("pca variance sum:", pca.explained_variance_ratio_.sum())

    with open(model_path, 'wb') as f:
        pickle.dump(pca, f, protocol=4)
        print("model saved in {}".format(model_path))
    return pca


if __name__ == "__main__":
    root = "/mnt/data3/caojh/dataset/AstroData"
    name = "trains_sets_correct"
    with open(os.path.join(os.path.join(root, "training"), name + ".pkl"), 'rb') as f:
        df = pickle.load(f)
        features = df.iloc[:, 0:2600].values
        model_path = os.path.join(os.path.join(root, "model"), "model_pca.pkl")
        train_pca(features, 750, model_path)
