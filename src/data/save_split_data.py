# Load Raw -- Split datasets to [ train, test, val ]

from src.data.load_data import load_raw_data
from src.data.preprocess import clean_text 
from src.data.split_data import split_dataset 


DATA_PATH = 'data/raw/twitter-raw-data.csv'

def main():
    df = load_raw_data(DATA_PATH)

    # basic clearn and split which can later useful for all ML / DL / Transformers 
    df['text'] = df['text'].apply(clean_text)

    train_df, val_df, test_df = split_dataset(df)

    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/val.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)

    print("Data pipeline completed Successfully..")


if __name__ == "__main__":
    main()

""" basic cleaning - to reusable for all ML / DL / Transforemers 
and advanced specific cleaning (preprocessing) will done during specific modeling...
"""