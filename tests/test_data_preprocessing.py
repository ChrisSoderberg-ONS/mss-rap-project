# tests/test_data_preprocessing.py
import pytest
import pandas as pd
from src.data_preprocessing import map_sex_column, drop_top_25_height

@pytest.fixture
def sample_df():
    data = {
        'Sex': ['M', 'F', 'I', 'M', 'F', 'I'],
        'Height': [0.5, 0.45, 0.55, 0.6, 0.4, 0.52],
        'Rings': [10, 15, 7, 8, 14, 9]
    }
    return pd.DataFrame(data)

def test_map_sex_column(sample_df):
    df = map_sex_column(sample_df.copy())
    expected_sex = [0, 1, 2, 0, 1, 2]
    assert df['Sex'].tolist() == expected_sex

def test_drop_top_25_height(sample_df):
    df = drop_top_25_height(sample_df.copy())
    assert len(df) == 4
    assert all(df['Height'] < sample_df['Height'].quantile(0.75))
