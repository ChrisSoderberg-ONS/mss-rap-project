"""
data_preprocessing.py

This module provides functions for preprocessing the Abalone dataset. It includes functions to map categorical
values, filter data, and split the dataset into training, validation, and test sets.
"""

import numpy as np

def map_sex_column(df):
    """
    Map the 'Sex' column in the DataFrame to numerical values.

    This function replaces the categorical 'Sex' values ('M', 'F', 'I') with numerical values 
    (0, 1, 2 respectively).

    Returns:
        pd.DataFrame: The DataFrame with the 'Sex' column mapped to numerical values.
    """
    # Map categorical 'Sex' values to numerical values
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    return df

def drop_top_25_height(df):
    """
    Drop the top 25% of rows based on the 'Height' column.

    This function calculates the 75th percentile of the 'Height' column and drops all rows
    where the 'Height' is greater than or equal to this value.

    Returns:
        pd.DataFrame: The DataFrame with the top 25% of 'Height' rows removed.
    """
    # Calculate the 75th percentile of the 'Height' column
    height_75_percentile = df['Height'].quantile(0.75)
    
    # Filter out rows where 'Height' is greater than or equal to the 75th percentile
    df = df[df['Height'] < height_75_percentile]
    return df

def split_data(df):
    """
    Split the DataFrame into training, validation, and test sets.

    This function shuffles the DataFrame and splits it into training (70%), validation (15%), 
    and test (15%) sets. It separates the features and target variable ('Rings').

    Returns:
        tuple: Six DataFrames corresponding to X_train, X_valid, X_test, y_train, y_valid, and y_test.
    """
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1)
    
    # Separate the features and target variable
    X = df_shuffled.drop("Rings", axis=1)
    y = df_shuffled["Rings"]
    
    # Calculate the split indices
    train_split = round(0.7 * len(df_shuffled))
    valid_split = round(train_split + 0.15 * len(df_shuffled))
    
    # Split the data into training, validation, and test sets
    X_train, y_train = X[:train_split], y[:train_split]
    X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
    X_test, y_test = X[valid_split:], y[valid_split:]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

