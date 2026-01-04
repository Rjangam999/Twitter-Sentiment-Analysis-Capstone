import keras
from keras import layers, Sequential, Input 
from keras import layers
# from keras.layers import Embedding, LSTM, Dense, Dropout , Bidirectional, BatchNormalization
from keras.optimizers import Adam 


def build_LSTM(
        vocab_size: int, 
        max_len: int, 
        embedding_dim: int = 128
):
    model = Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len), 
        layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.3)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')
    ])

    model.build(input_shape=(None, max_len))

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model 