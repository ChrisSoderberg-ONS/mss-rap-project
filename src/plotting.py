import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_matrix(df):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
    for i, col in enumerate(df.columns):
        sns.regplot(x=col, y="Rings", data=df, ax=axes[i // 3, i % 3], ci=None, scatter_kws={'alpha': 0.5})
        axes[i // 3, i % 3].set_title(col)
    plt.tight_layout()
    plt.show()

def plot_height_boxplot(df):
    plt.boxplot(df['Height'], vert=False)
    plt.show()

