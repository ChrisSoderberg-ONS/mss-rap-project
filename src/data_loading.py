"""
data_loading.py

This module provides a function to load and preprocess data from the UCI Machine Learning Repository.
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

def load_data():
    """
    Load and preprocess the Abalone dataset from the UCI Machine Learning Repository.

    This function fetches the Abalone dataset (id=1) from the UCI Machine Learning Repository
    and adds a 'Rings' column as the target feature.

    Returns:
        pd.DataFrame: A DataFrame containing the Abalone dataset with features and the target 'Rings' column.
    """
    # Fetch the Abalone dataset from the UCI Machine Learning Repository
    abalones = fetch_ucirepo(id=1)
    
    # Convert the features data to a pandas DataFrame
    df = pd.DataFrame(abalones.data.features)
    
    # Add the target values as a new column named 'Rings'
    df['Rings'] = abalones.data.targets
    
    return df

