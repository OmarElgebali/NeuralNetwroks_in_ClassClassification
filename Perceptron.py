from random import random
import tkinter as tk
from tkinter import W
from tkinter import *
import numpy as np
import pandas as pd
from NN_Main import boxs, radio, LearningRate, Epochs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def takePara():
def readFile():
    df = pd.read_csv('Dry_Bean_Dataset.csv')
    if radio.get() == 1:
        select_row = df.iloc[0:101]
    if radio.get() == 2:
        select_row = pd.concat(df.iloc[0:51], df.iloc[100:151])
    elif radio.get() == 3:
        select_row = df.iloc[50:151]
    return select_row


# feature1, feature2 = boxs.get(boxs.curselection())
selected_indices = boxs.curselection()

feature1 = boxs.get(selected_indices[0])
feature2 = boxs.get(selected_indices[1])


def getfeature(feat1, feat2):
    columns = readFile()
    le = LabelEncoder()
    label = le.fit_transform(columns['Class'])
    label = [i if i != 0 else -1 for i in label]
    feature = columns[[feat1, feat2]]
    label = columns['Class']
    return feature, label


features, target_class = getfeature(feature1, feature2)


def train_test(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=42)
    feature1_train = X_train[0]
    feature2_train = X_train[1]
    feature1_test = X_test[0]
    feature2_test = X_test[1]
    return feature1_train, feature2_train, feature1_test, feature2_test


trainfeat1, trainfeat2, testfeat1, testfeat2 = train_test()


def preprocessing(trainf1, trainf2, testf1, testf2):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
    normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
    normF1_test = scaler.fit_transform(np.array(testf1).reshape(-1, 1))
    normF2_test = scaler.fit_transform(np.array(testf2).reshape(-1, 1))

    return normF1_train, normF2_train, normF1_test, normF2_test


normfeat1_train, normfeat2_train, normfeat1_test, normfeat2_test = preprocessing(trainfeat1, trainfeat2, testfeat1,
                                                                                 testfeat2)

learning_rate = float(LearningRate.get())
epoc = int(Epochs.get())


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

    return w0, w1, w2

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
# w0, w1, w2 = perceptron(normfeat1_train, normfeat2_train, target_class, learning_rate, epoc)
#
# print("W0", w0, "W1", w1, "W2", w2)
