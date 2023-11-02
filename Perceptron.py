from random import random
import tkinter as tk
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

feature = df[['Area', 'Perimeter']]
label = df['Class']

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=42)

# train_data = pd.concat([class1_train, class2_train])
# test_data = pd.concat([class1_test, class2_test])

# X_train = train_data[['Area', 'Perimeter']]
# y_train = train_data['Class']
print("X_train shape :", X_train.shape)
print("y_train shape :", y_train.shape)
# X_test = test_data[['Area', 'Perimeter']]
# y_test = test_data['Class']
print("X_train shape :", X_test.shape)
print("y_train shape :", y_test.shape)
print(y_test)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_train = [i if i != 0 else -1 for i in y_train]
y_test = le.fit_transform(y_test)
y_test = [i if i != 0 else -1 for i in y_test]
# print(y_train)
print(y_test)
feature1_train = X_train['Area']
feature2_train = X_train['Perimeter']
feature1_test = X_test['Area']
feature2_test = X_test['Perimeter']
# feature_range=(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
normF1_train = scaler.fit_transform(np.array(feature1_train).reshape(-1, 1))
normF2_train = scaler.fit_transform(np.array(feature2_train).reshape(-1, 1))
normF1_test = scaler.fit_transform(np.array(feature1_test).reshape(-1, 1))
normF2_test = scaler.fit_transform(np.array(feature2_test).reshape(-1, 1))
print("f1 train", normF1_train)
print("f1 test", normF1_test)


def perceptron(feature, feature1, target, learning_rate):
    def activation(y):
        return 1 if y >= 0 else -1

    w0 = 0.2
    w1 = 0.867
    w2 = 0.75011
    while True:
        errorCute = 0
        for x1, x2, t in zip(feature, feature1, target):
            y = x1 * w1 + x2 * w2 + w0
            yk = activation(y)
            error = t - yk

            if error != 0:
                errorCute += 1

            w0 = w0 + learning_rate * error
            w1 = w1 + learning_rate * error * float(x1)
            w2 = w2 + learning_rate * error * float(x2)
        if errorCute == 0:
            break

    return w0, w1, w2


w0, w1, w2 = perceptron(normF1_train, normF2_train, y_train, 0.5)
print(w0, w1, w2)


def Bigteste():
    def activation(y):
        return 1 if y >= 0 else -1

    for i in range(40):
        x1 = normF1_test[i]
        x2 = normF2_test[i]
        yk = y_test[i]
        y = x1 * w1 + x2 * w2 + w0
        y = activation(y)
        print("Predicted", y)
        print("Acutal", yk)


Bigteste()
def Ibrahim_Insisted():
    main = tk.Tk()
    boxs = tk.Listbox(main)
    boxs.grid(row=0, column=2)
    boxs.insert(0, 'Area')
    boxs.insert(1, 'Perimeter')
    boxs.insert(2, 'MajorAxisLength')
    boxs.insert(3, 'MinorAxisLength')
    boxs.insert(4, 'roundnes')

    main.title('Perceptorn & Adaline')

    main.mainloop()

