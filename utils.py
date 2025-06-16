import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import umap.umap_ as umap
from gower import gower_matrix
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.spatial.distance import hamming
from itertools import combinations
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.pipeline import Pipeline
from kmodes.kprototypes import KPrototypes


class KPrototypesClusters(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, init='Huang', gamma=0.06, n_jobs=-1, random_state=42):
        self.n_clusters = n_clusters
        self.init = init
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = None

    def fit(self, X, categorical=None):
        self.model = KPrototypes(
            n_clusters=self.n_clusters,
            init=self.init,
            gamma=self.gamma,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        self.model.fit(X, categorical=categorical)
        return self

    def predict(self, X, categorical=None):
        if self.model is None:
            raise RuntimeError("You must fit the model before calling predict.")
        return self.model.predict(X, categorical=categorical)


def multiple_graphs(n_cols: int, target: str, data: pd.DataFrame):
    rows = (len(data.columns) + n_cols - 1) // n_cols
    plt.figure(figsize=(18, rows * 5))
    for i, f in enumerate(data.drop(target, axis=1).columns.to_list(), 1):
        plt.subplot(rows, n_cols, i)
        sns.lineplot(x=target, y=f, data=data)
    plt.show()


# Расстояние между кластерами (Inter-Cluster Distance)
# Показывает, насколько кластеры разделены. Чем больше расстояние, тем лучше.
def inter_cluster_distance(centroids):
    distances = []
    for (c1, c2) in combinations(centroids, 2):
        distances.append(hamming(c1, c2))
    return np.mean(distances)


def draw_clusters(dataframe: pd.DataFrame):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(dataframe)
    reducer = umap.UMAP()
    _emb = reducer.fit_transform(encoded)
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        x=_emb[:, 0],
        y=_emb[:, 1],
        alpha=0.7
    )
    plt.title('UMAP')
    plt.tight_layout()
    plt.show()


def scores_calc(dataframe, clrs, metric):
    gm = gower_matrix(dataframe)
    shi = silhouette_score(gm, clrs, metric=metric)
    chi = calinski_harabasz_score(gm, clrs)
    dbi = davies_bouldin_score(gm, clrs)
    return shi, chi, dbi





