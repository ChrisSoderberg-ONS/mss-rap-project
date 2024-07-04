"""
plotting.py

This module provides functions to visualize data using scatter matrix plots and boxplots.
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_matrix(df):
    """
    Plot a scatter matrix of all columns in the DataFrame against the 'Rings' column.

    This function creates scatter plots with regression lines for each feature in the DataFrame
    against the 'Rings' target column, organized in a 4x3 grid.

    Args:
        df: The input DataFrame containing the features and 'Rings' target column.
    """
    # Create a figure with a 4x3 grid of subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))
    
    # Loop through each column and plot a scatter plot with a regression line
    for i, col in enumerate(df.columns):
        sns.regplot(x=col, y="Rings", data=df, ax=axes[i // 3, i % 3], ci=None, scatter_kws={'alpha': 0.5})
        axes[i // 3, i % 3].set_title(col)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plot
    plt.show()

def plot_height_boxplot(df):
    """
    Plot a horizontal boxplot of the 'Height' column in the DataFrame.

    This function creates a boxplot to visualize the distribution of the 'Height' column.

    Args:
        df: The input DataFrame containing the 'Height' column.
    """
    # Create a horizontal boxplot for the 'Height' column
    plt.boxplot(df['Height'], vert=False)
    
    # Display the plot
    plt.show()


