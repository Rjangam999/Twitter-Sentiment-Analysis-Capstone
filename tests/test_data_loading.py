from src.data.load_data import load_raw_data 
import pandas as pd

def test_load_raw_data_structure():
    df = load_raw_data("data/raw/twitter-raw-data.csv")

    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns
    assert "target" in df.columns