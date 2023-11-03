from random import random
import tkinter as tk
from tkinter import W
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def readFile(radio):
    df = pd.read_csv('Dry_Bean_Dataset.csv')
    if radio == 1:
        select_row = df.iloc[0:101]
    if radio == 2:
        select_row = pd.concat(df.iloc[0:51], df.iloc[100:151])
    elif radio == 3:
        select_row = df.iloc[50:151]
    return select_row


# feature1, feature2 = boxs.get(boxs.curselection())

def getfeature(feat1, feat2, col):
    le = LabelEncoder()
    label = le.fit_transform(col['Class'])
    label = [i if i != 0 else -1 for i in label]
    feature = col[[feat1, feat2]]
    label = col['Class']
    return feature, label


def train_test(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=22)
    feature1_train = X_train[0]
    feature2_train = X_train[1]
    feature1_test = X_test[0]
    feature2_test = X_test[1]
    return feature1_train, feature2_train, feature1_test, feature2_test



def preprocessing(trainf1, trainf2, testf1, testf2):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
    normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
    normF1_test = scaler.fit_transform(np.array(testf1).reshape(-1, 1))
    normF2_test = scaler.fit_transform(np.array(testf2).reshape(-1, 1))

    return normF1_train, normF2_train, normF1_test, normF2_test


def perceptron_train(feat1Train, feat2train, T_class, learning, epoch):
    def activation(y):
        return 1 if y >= 0 else -1

    w0 = 0.2
    w1 = 0.867
    w2 = 0.75011
    for k in range(epoch):
        errorCute = 0
        for x1, x2, t in zip(feat1Train, feat2train, T_class):
            y = x1 * w1 + x2 * w2 + w0
            yk = activation(y)
            error = t - yk

            if error != 0:
                errorCute += 1

            w0 = w0 + learning * error
            w1 = w1 + learning * error * float(x1)
            w2 = w2 + learning * error * float(x2)
        if errorCute == 0:
            break
    print(w0, w1, w2)
    return w0, w1, w2


def big():
    featuring = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']
    selected_indices = boxs.curselection()
    print(selected_indices)
    selected_features = [featuring[i] for i in selected_indices]
    boxs1 = boxs.get(selected_indices[0])
    print(boxs1)
    boxs2 = boxs.get(selected_indices[1])
    print(boxs2)
    radioo = int(radio.get())
    Learning_Rate = float(LearningRate.get())
    Epochss = int(Epochs.get())

    columns = readFile(radioo)
    feature1 = boxs1
    feature2 = boxs2
    features, target_class = getfeature(feature1, feature2, columns)
    trainfeat1, trainfeat2, testfeat1, testfeat2 = train_test(features, target_class)
    normfeat1_train, normfeat2_train, normfeat1_test, normfeat2_test = preprocessing(trainfeat1, trainfeat2, testfeat1, testfeat2)
    learning_rate = Learning_Rate
    epoc = Epochss
    w0, w1, w2 = perceptron_train(normfeat1_train, normfeat2_train, target_class, learning_rate, epoc)
    print(w0, w1, w2)


# w0, w1, w2 = perceptron(normF1_train, normF2_train, y_train, 0.5)
# print(w0, w1, w2)
#
#
# def Bigteste():
#     def activation(y):
#         return 1 if y >= 0 else -1
#
#     for i in range(40):
#         x1 = normF1_test[i]
#         x2 = normF2_test[i]
#         yk = y_test[i]
#         y = x1 * w1 + x2 * w2 + w0
#         y = activation(y)
#         print("Predicted", y)
#         print("Acutal", yk)
#
#
# Bigteste()
#
# print("W0", w0, "W1", w1, "W2", w2)
import tkinter as tk
main = tk.Tk()
main.title('Perceptorn & Adaline')
boxs = tk.Listbox(main, selectmode=tk.MULTIPLE)
boxs.pack()
boxs.insert(0, 'Area')
boxs.insert(1, 'Perimeter')
boxs.insert(2, 'MajorAxisLength')
boxs.insert(3, 'MinorAxisLength')
boxs.insert(4, 'roundnes')


radio = tk.IntVar()
tk.Radiobutton(main, text="C1 & C2", variable=radio, value=1).pack()
tk.Radiobutton(main, text="C1 & C2", variable=radio, value=2).pack()
tk.Radiobutton(main, text="C3 & C2", variable=radio, value=3).pack()

tk.Label(main, text='Learning Rate').pack()
LearningRate = tk.Entry(main)
LearningRate.pack()

options = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes']


# var = tk.StringVar()
# cb = tk.Checkbutton(main, text=options[0], variable= 'Area', offvalue="")
# cb.deselect()
# cb.pack()
#
# var1 = tk.StringVar()
# cb1 = tk.Checkbutton(main, text=options[1], variable=var1, offvalue="")
# cb1.deselect()
# cb1.pack()

tk.Label(main, text='Epochs').pack()
Epochs = tk.Entry(main)
Epochs.pack()


button = tk.Button(main, text="Perceptron", width=10, height=3, command=lambda: big())
button.pack()

main.mainloop()

