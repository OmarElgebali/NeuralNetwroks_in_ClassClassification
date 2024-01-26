import keras.layers
from gensim.models import Word2Vec
import numpy as np

# Provided data
data = [
    ("شرك زبال سواق تبرشم مفيش  حاجز قبل يوم", -1),
    ("خدم دفع طريق كي نت توقف عند صبح قط دفع قدا", 1),
    (" كتر مرة  يكون خصوم قو على وجب متاح", -1),
    ("عل طبيق ممتاز لم سر دون رجوع لى خدم عملاء", 1),
]


def input_embedding(data, vector_dimension):
    # Tokenize the text
    tokenized_data = [review.split() for review, _ in data]
    # tokenized_data: array of all words
    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_data, vector_size=vector_dimension, window=5, min_count=1, workers=4)

    # Function to convert a word to its vector
    def word_to_vector(word, model):
        if word in model.wv:
            return model.wv[word]
        else:
            return np.zeros(vector_dimension)

    # Transform the data to feature vectors
    X_vectors = []
    for review in tokenized_data:
        for word in review:
            X_vectors.append(word_to_vector(word, word2vec_model))

    return X_vectors


# =======

def positional_encoding_for_vectors(word_vectors, vector_dimension):
    word_vectors = np.array(word_vectors)

    max_len, _ = word_vectors.shape
    position = np.arange(0, max_len, dtype=np.float32)
    # iv_term = np.exp(np.arange(0, vector_dimension, 2, dtype=np.float32) * -(np.log(10000.0) / vector_dimension))
    div_term = np.exp(
        np.arange(0, vector_dimension, 2, dtype=np.float32) * -(np.log(10000.0) / vector_dimension)).ravel()
    pos_enc = np.zeros((max_len, vector_dimension))
    pos_enc[:, 0::2] = np.sin(position[:, np.newaxis] * div_term)
    pos_enc[:, 1::2] = np.cos(position[:, np.newaxis] * div_term)

    return word_vectors + pos_enc


# Example word vectors for each word in the sentence

# Set the dimensionality of the model (you can adjust this as needed)
word_vectors = input_embedding(data, 8)
print("word_vectors", word_vectors)
# Apply positional encoding to word vectors

# must enter even vector_dimension
word_vectors_with_position = positional_encoding_for_vectors(word_vectors, 8)
print("word_vectors_with_position", word_vectors_with_position)

# =======
