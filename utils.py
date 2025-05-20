import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def multiple_graphs(n_cols: int, target: str, data: pd.DataFrame):
    rows = (len(data.columns) + n_cols - 1) // n_cols
    plt.figure(figsize=(18, rows * 5))
    for i, f in enumerate(data.drop(target, axis=1).columns.to_list(), 1):
        plt.subplot(rows, n_cols, i)
        sns.lineplot(x=target, y=f, data=data)
    plt.show()