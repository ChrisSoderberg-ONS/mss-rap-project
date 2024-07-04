"""
test_data_preprocessing.py

This module contains tests for the functions in the data_preprocessing.py module.
"""

import pytest
import pandas as pd
from src.data_preprocessing import map_sex_column, drop_top_25_height

@pytest.fixture
def sample_preprocessing_df():
    """
    Fixture that provides a sample DataFrame for preprocessing tests.

    Returns:
        data: A DataFrame with 'Sex', 'Height', and 'Rings' columns.
    """
    data = {
        'Sex': ['M', 'F', 'I', 'M', 'F', 'I'],
        'Height': [0.5, 0.45, 0.55, 0.6, 0.4, 0.52],
        'Rings': [10, 15, 7, 8, 14, 9]
    }
    return pd.DataFrame(data)

def test_map_sex_column(sample_preprocessing_df):
    """
    Test the map_sex_column function 
    to ensure it correctly maps 'Sex' column values to numerical values.

    Args:
        sample_preprocessing_df: The sample DataFrame provided by the fixture.
    """
    df = map_sex_column(sample_preprocessing_df.copy())
    expected_sex = [0, 1, 2, 0, 1, 2]
    assert df['Sex'].tolist() == expected_sex

def test_drop_top_25_height(sample_preprocessing_df):
    """
    Test the drop_top_25_height function to ensure it correctly drops the top 25% of rows based on 'Height'.

    Args:
        sample_preprocessing_df: The sample DataFrame provided by the fixture.
    """
    df = drop_top_25_height(sample_preprocessing_df.copy())
    
    # Ensure the resulting DataFrame has the correct number of rows
    assert len(df) == 4
    
    # Ensure all remaining heights are below the 75th percentile of the original heights
    assert all(df['Height'] < sample_preprocessing_df['Height'].quantile(0.75))

