import os
import re
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, f1_score, accuracy_score, recall_score, roc_auc_score, \
    make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score

# from tashaphyne.stemming import ArabicLightStemmer

df1 = pd.read_csv('Train_Dataset_After_Preprocessing.csv')

x_train = df1['review_description']
y_train = df1['rating']

df2 = pd.read_csv('Test_Dataset_Labeled_After_Preprocessing.csv')
df2 = df2.dropna()
y_test = df2['rating']
x_test = df2['review_description']
test_ids = df2['ID']

vocab_size = 18000  # choose based on statistics
oov_tok = ''
max_length = 100  # choose based on statistics, for example 150 to 200
padding_type = 'post'
trunc_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(x_train)
word_index = tokenizer.word_index
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)

x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

from keras.utils import to_categorical

y_train = y_train + 1  # Shift to start from 0, making -1 to 0, 0 to 1, 1 to 2
y_train = to_categorical(y_train, num_classes=3)

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


# Define a function to create your Keras model
def create_model(dropout_factor=0.5, n_units=32, embedding_dim=10, max_length=100):
    model = Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(LSTM(n_units)),
        keras.layers.Dropout(dropout_factor),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dropout(dropout_factor),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


num_epochs = 3
model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=1, dropout_factor=0.5, n_units=32,
                        embedding_dim=10, max_length=100)

param_grid = {
    'dropout_factor': [0.3, 0.5, 0.7],
    'n_units': [16, 32, 64],
    'embedding_dim': [5, 10, 15],
    'max_length': [20, 80, 100, 120],
    'epochs': [2, 3, 4, 5, 10, 15]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

best_params = grid_result.best_params_
print("BiLSTM - Best Parameters:", best_params)

best_score = grid_result.best_score_
print("BiLSTM - Best Score:", best_score)


dropout_factor = best_params['dropout_factor']
n_units = best_params['n_units']
embedding_dim = best_params['embedding_dim']
max_length = best_params['max_length']
num_epochs = best_params['epochs']

model_name = f'model_LSTM_nUnits{n_units}_drop{dropout_factor}_epoch{num_epochs}_empdim{embedding_dim}_maxlen{max_length}.tfl'

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

print("BiLSTM - Accuracy of prediction on test set : ", accuracy_score(y_test, decoded_predictions))
