import pandas as pd
from src.data_loading import load_data

def test_load_data():
    df = load_data()
    assert isinstance(df, pd.DataFrame), "The data should be loaded into a pandas DataFrame"
    assert 'Rings' in df.columns, "The DataFrame should contain a 'Rings' column"
