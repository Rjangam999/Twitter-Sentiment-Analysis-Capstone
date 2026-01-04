import keras 
from keras.layers import TextVectorization
from keras.utils import pad_sequences

def build_and_train_tokenizer(texts, vocab_size=20000, max_len=100):
    # initialize the layer 
    vectorizer = TextVectorization(
        max_tokens=vocab_size, 
        output_mode='int', 
        output_sequence_length=100
    )
    # tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    # tokenizer.fit_on_texts(texts) ----- old method 
    vectorizer.adapt(texts)

    return vectorizer 

# def tokenize_and_pad(tokenizer, texts, max_len):
#     sequences = tokenizer.texts_to_sequences(texts)
#     padded = pad_sequences(sequences, max_len=max_len, padding='post', truncating='post')
#     return padded             ---- old method   


def tokenize_and_pad(vectorizer, texts):
    padded = vectorizer(texts)
    return padded 

# # Transform text to padded sequences (-- Tensor ready for model )
# padded_sequences = vectorizer(texts)