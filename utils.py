import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.spatial.distance import hamming
from itertools import combinations


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




