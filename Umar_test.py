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
        select_row = df.iloc[0:102]
    if radio == 2:
        half1 = df.iloc[:50]
        half2 = df.iloc[100:]
        select_row = pd.concat([half1, half2])
    elif radio == 3:
        select_row = df.iloc[51:150]
    return select_row


def get_feature(feat1, feat2, croppedData):
    feature = croppedData[[feat1, feat2]]
    label = croppedData['Class']
    return feature, label


def train_test(feature, label, f1, f2):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, stratify=label, random_state=22)
    feature1_train = X_train[f1]
    feature2_train = X_train[f2]
    feature1_test = X_test[f1]
    feature2_test = X_test[f2]
    targetTrain = y_train
    targetTest = y_test
    return feature1_train, feature2_train, feature1_test, feature2_test, targetTrain, targetTest


def preprocessing(trainf1, trainf2, testf1, testf2, trainClass, testClass):
    le = LabelEncoder()
    # print(trainClass)
    encodedTrainC = le.fit_transform(trainClass)
    # print(encodedTrainC)
    encodedTrainC = [i if i != 0 else -1 for i in encodedTrainC]
    # print(testClass)
    encodedTestC = le.fit_transform(testClass)
    # print(encodedTestC)
    encodedTestC = [i if i != 0 else -1 for i in encodedTestC]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
    normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
    normF1_test = scaler.fit_transform(np.array(testf1).reshape(-1, 1))
    normF2_test = scaler.fit_transform(np.array(testf2).reshape(-1, 1))

    return normF1_train, normF2_train, normF1_test, normF2_test, encodedTrainC, encodedTestC


# [0 0 1 1 0 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 0 0 1 1 1 0 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 1 0 0 0 1 1]
# [B B S S B B B S B S S B S S S S S S B B S S S B B B B S B B S S S B S B S S B S S B B B B S B B S S B B S B S B B B S S]
# [0 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 0 1 0 1]
# [B S B S S B B B S B S B S S B S S S B S B B S B B B S S S B B S S S B B B S B S]


def perceptron_train(feat1Train, feat2train, T_class, learning, epoch):
    def activation(y):
        return 1 if y >= 0 else -1

    w0 = 0.2
    w1 = 0.867
    w2 = 0.75011
    for k in range(epoch):
        # while True:
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
    w = [w0, w1, w2]
    return w


def Bigteste(testsample1, testsample2, testclass, weight):
    def activation(y):
        return 1 if y >= 0 else -1

    for i in range(40):
        x1 = testsample1[i]
        x2 = testsample2[i]
        yk = testclass[i]
        y = x1 * weight[1] + x2 * weight[2] + weight[0]
        y = activation(y)
        print("Predicted", y, "Actual", yk)
        # print("Acutal", yk)


# 0.19999999999999996 -1.0192994731258205 -1.0819145688874579
def execute(chunk, feat1, feat2, lR, epch):
    reading = readFile(chunk)
    twoFeatures, targetClass = get_feature(feat1, feat2, reading)
    train1, train2, test1, test2, trainClassSample, testClassSample = train_test(twoFeatures, targetClass, feat1,
                                                                                 feat2)
    proF1train, proF2train, prof1test, prof2test, ClassTrain, ClassTest = preprocessing(train1, train2, test1, test2
                                                                                        , trainClassSample,
                                                                                        testClassSample)

    weights = perceptron_train(proF1train, proF2train, ClassTrain, lR, epch)
    Bigteste(prof1test, prof2test, ClassTest, weights)

# execute(3, 'Area', 'Perimeter', 0.5, 100)
