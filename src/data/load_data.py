import pandas as pd 
import os 

def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns=['target', 'id', 'date', 'flag', 'user', 'text']
    df['target'] = df['target'].map({0:'negative', 4:'positive'})
    return df[['text', 'target']]


def sample_data_path() -> str:
    """ Return default sample path """
    return os.path.join(os.getcwd(), 'raw', 'twitter-raw-data.csv')