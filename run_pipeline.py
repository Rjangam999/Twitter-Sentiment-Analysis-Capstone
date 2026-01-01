from src.data.load_data import load_raw_data 
from src.data.preprocess import clean_text 

DATA_PATH = 'data/raw/twitter-raw-data.csv'

def main():
    df = load_raw_data(DATA_PATH)
    df['text'] = df['text'].apply(clean_text)
    print(df.head())


if __name__ == "__main__":
    main()