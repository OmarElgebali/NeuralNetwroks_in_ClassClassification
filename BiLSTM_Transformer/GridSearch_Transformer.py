from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df1 = pd.read_csv('Train_Dataset_After_Preprocessing.csv')

x_train = df1['review_description']
y_train = df1['rating']

df2 = pd.read_csv('Test_Dataset_Labeled_After_Preprocessing.csv')
df2 = df2.dropna()
x_test = df2['review_description']
test_ids = df2['ID']
y_test = df2['rating']


# ==========
oov_tok = ''

vocab_size = 70000
max_length = 90
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(x_train)
x_train = keras.utils.pad_sequences(x_train, maxlen=max_length)

x_test = tokenizer.texts_to_sequences(x_test)
x_test = keras.utils.pad_sequences(x_test, maxlen=max_length)

y_train = y_train + 1  # Shift to start from 0, making -1 to 0, 0 to 1, 1 to 2

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# -=========
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# Function to create your Keras Transformer model
def create_transformer_model(num_heads=2, ff_dim=32, dropout_factor=0.1, embed_dim=32, num_epochs=2):
    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_factor)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(dropout_factor)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


param_grid = {
    'num_heads': [2, 4],
    'ff_dim': [16, 32, 64],
    'dropout_factor': [0.1, 0.2],
    'embed_dim': [32, 64, 128],
    'num_epochs': [2, 3, 4]
}

# Create KerasClassifier with the custom function
model = KerasClassifier(build_fn=create_transformer_model, verbose=1, num_heads=2, ff_dim=32, dropout_factor=0.1, embed_dim=32, num_epochs=2)

# Create GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)

# Perform grid search
grid_result = grid.fit(x_train, y_train)

best_params = grid_result.best_params_
print("Best Parameters:", best_params)

best_score = grid_result.best_score_
print("Best Score:", best_score)

# best_model = grid_result.best_estimator_.model

num_heads = best_params['num_heads']
ff_dim = best_params['ff_dim']
dropout_factor = best_params['dropout_factor']
embed_dim = best_params['embed_dim']
num_epochs = best_params['num_epochs']

model_name = f'model_transformer_num_heads{num_heads}_ff_dim{ff_dim}_dropout_factor{dropout_factor}_epoch{num_epochs}_empdim{embed_dim}_maxlen{max_length}_vocab_size{vocab_size}.tfl'


inputs = layers.Input(shape=(max_length,))
embedding_layer = TokenAndPositionEmbedding(max_length, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(dropout_factor)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(dropout_factor)(x)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=num_epochs, validation_split=0.1)
model.save(model_name)

predictions = model.predict(x_test)
decoded_predictions = np.argmax(predictions, axis=1) - 1
print(f'## {decoded_predictions} ##')
print(f'## {set(decoded_predictions)} ##')
from collections import Counter
value_counts = Counter(decoded_predictions)
for value, count in value_counts.items():
    print(f"## {value}: {count} times ##")
result_df = pd.DataFrame({
    'ID': test_ids,
    'rating': decoded_predictions
})

result_df.to_csv('Team_5_TRANSFORMER_RESULTS_4.csv', index=False)
print("Accuracy of prediction on test set : ", accuracy_score(y_test, decoded_predictions))
