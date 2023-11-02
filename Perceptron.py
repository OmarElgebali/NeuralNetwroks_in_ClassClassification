from random import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

csv_file_path = 'cropped_file_P.csv'

df = pd.read_csv(csv_file_path)

# Extract the first two columns and the last column
selected_columns = df.iloc[:, [0, 1, -1]]

# lable = df['Class']
le = LabelEncoder()
label = le.fit_transform(df['Class'])
label = [i if i != 0 else -1 for i in label]
# print(label)
# weight0 = random()
# weight1 = random()
# weight2 = random()
weight0 = 0.2
weight1 = 0.867
weight2 = 0.75011

we = []
we.append(weight0)
we.append(weight1)
we.append(weight2)

feature = df['Area', 'Perimeter']
# feature2 = df['Perimeter']

# feature[0] = np.array(feature[0]).reshape(-1, 1)
# feature[1] = np.array(feature[1]).reshape(-1, 1)

# normF2 = scaler.fit_transform(feature2)
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=42)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(feature)
X_test = scaler.transform(X_test)
# def perceptron(feature, feature1, w, target, learning_rate):
#     def activation(y):
#         return 1 if y >= 0 else -1
#
#     while True:
#         errorCute = 0
#         for x1, x2, t in zip(feature, feature1, target):
#             y = x1 * w[1] + x2 * w[2] + w[0]
#             yk = activation(y)
#             error = t - yk
#
#             if error != 0:
#                 errorCute += 1
#
#             w[0] = w[0] + learning_rate * error
#             w[1] = w[1] + learning_rate * error * x1
#             w[2] = w[2] + learning_rate * error * x2
#         if errorCute == 0:
#             break
#
#     return w
#
#
# print(f' w : {perceptron(normF1, normF2, we, label, 0.15)}')
