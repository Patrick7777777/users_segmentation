import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from itertools import combinations
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform


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


def mixed_metrics(dataframe, clusters, categorical_cols, numerical_cols, gamma='auto'):
    """
    gamma='auto' - вычисляет gamma как в KPrototypes
    gamma=float - заданное значение
    """
    # Конвертация в индексы
    cat_indices = [dataframe.columns.get_loc(col) for col in categorical_cols]
    num_indices = [dataframe.columns.get_loc(col) for col in numerical_cols]

    # Кодирование категориальных признаков
    df_encoded = dataframe.copy()
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    # Автоматический расчет gamma
    if gamma == 'auto':
        num_std = np.mean(np.std(df_encoded[numerical_cols], axis=0))
        # num_std = np.mean(np.std(StandardScaler().fit_transform(df_encoded[numerical_cols]), axis=0))
        cat_std = np.mean(np.std(df_encoded[categorical_cols], axis=0))
        gamma = num_std / (cat_std + 1e-8)

    # Комбинированное расстояние
    def mixed_distance(x, y):
        num_dist = np.linalg.norm(x[num_indices] - y[num_indices])
        cat_dist = np.mean(x[cat_indices] != y[cat_indices])
        return num_dist + gamma * cat_dist

    # Матрица расстояний и метрики
    x_array = df_encoded.values
    dist_matrix = squareform(pdist(x_array, lambda u, v: mixed_distance(u, v)))

    shi = silhouette_score(dist_matrix, clusters, metric='precomputed')
    chi = calinski_harabasz_score(df_encoded.values, clusters)
    dbi = davies_bouldin_score(df_encoded.values, clusters)

    return shi, chi, dbi, gamma  # Возвращаем использованное gamma



