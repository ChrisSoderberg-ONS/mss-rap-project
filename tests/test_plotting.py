import pandas as pd
import matplotlib.pyplot as plt
from src.plotting import plot_scatter_matrix, plot_height_boxplot

def test_plot_scatter_matrix():
    df = pd.DataFrame({
        'Length': [0.1, 0.2, 0.3],
        'Diameter': [0.1, 0.2, 0.3],
        'Height': [0.1, 0.2, 0.3],
        'Whole weight': [0.1, 0.2, 0.3],
        'Shucked weight': [0.1, 0.2, 0.3],
        'Viscera weight': [0.1, 0.2, 0.3],
        'Shell weight': [0.1, 0.2, 0.3],
        'Rings': [1, 2, 3]
    })
    plot_scatter_matrix(df)
    plt.close()

def test_plot_height_boxplot():
    df = pd.DataFrame({'Height': [0.1, 0.2, 0.3, 0.4]})
    plot_height_boxplot(df)
    plt.close()
