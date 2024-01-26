import os
import keras
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


df1 = pd.read_csv('Train_Dataset_After_Preprocessing.csv')

x_train = df1['review_description']
y_train = df1['rating']

df2 = pd.read_csv('Test_Dataset_NoLabel_After_Preprocessing.csv')
x_test = df2['review_description']
test_ids = df2['ID']

vocab_size = 18000
oov_tok = ''
embedding_dim = 10
max_length = 100
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)

x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

y_train = y_train + 1
y_train = to_categorical(y_train, num_classes=3)


dropout_factor = 0.5
n_units = 32
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Bidirectional(keras.layers.LSTM(n_units)),
    keras.layers.Dropout(dropout_factor),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dropout(dropout_factor),
    keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

num_epochs = 2

model_name = f'model_BiLSTM_nUnits{n_units}_drop{dropout_factor}_epoch{num_epochs}_empdim{embedding_dim}_maxlen{max_length}_vocab_size{vocab_size}.tfl'
# model_name = f'model_BiLSTM_nUnits32_drop0.5_epoch2_empdim10_maxlen100_vocab_size18000.tfl'
if os.path.exists(model_name):
    model = keras.models.load_model(f'./{model_name}')
else:
    history = model.fit(x_train, y_train,
                        epochs=num_epochs, verbose=1,
                        validation_split=0.1)
    model.save(model_name)


predictions = model.predict(x_test)
decoded_predictions = np.argmax(predictions, axis=1) - 1

result_df = pd.DataFrame({
    'ID': test_ids,
    'rating': decoded_predictions
})

result_df.to_csv('Team_5_LSTM_RESULTS.csv', index=False)
