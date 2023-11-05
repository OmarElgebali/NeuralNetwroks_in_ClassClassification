from random import random
import tkinter as tk
from tkinter import W
from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, stratify=label, random_state=22)
    feature1_train = X_train[f1]
    feature2_train = X_train[f2]
    feature1_test = X_test[f1]
    feature2_test = X_test[f2]
    targetTrain = y_train
    targetTest = y_test
    return feature1_train, feature2_train, feature1_test, feature2_test, targetTrain, targetTest


# adalin 0,1
def preprocessing(trainf1, trainf2, testf1, testf2, trainClass, testClass):
    le = LabelEncoder()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    encodedTrainC = le.fit_transform(trainClass)
    encodedTrainC = [i if i != 0 else -1 for i in encodedTrainC]
    normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
    normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
    encodedTestC = le.fit_transform(testClass)
    encodedTestC = [i if i != 0 else -1 for i in encodedTestC]
    trainf1 = trainf1.fillna(trainf1.mean())
    trainf2 = trainf2.fillna(trainf2.mean())
    testf1 = testf1.fillna(testf1.mean())
    testf2 = testf2.fillna(testf2.mean())
    normF1_test = scaler.transform(np.array(testf1).reshape(-1, 1))
    normF2_test = scaler.transform(np.array(testf2).reshape(-1, 1))

    return normF1_train, normF2_train, normF1_test, normF2_test, encodedTrainC, encodedTestC


# [0 0 1 1 0 0 0 1 0 1 1 0 1 1 1 1 1 1 0 0 1 1 1 0 0 0 0 1 0 0 1 1 1 0 1 0 1 1 0 1 1 0 0 0 0 1 0 0 1 1 0 0 1 0 1 0 0 0 1 1]
# [B B S S B B B S B S S B S S S S S S B B S S S B B B B S B B S S S B S B S S B S S B B B B S B B S S B B S B S B B B S S]
# [0 1 0 1 1 0 0 0 1 0 1 0 1 1 0 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 0 1 1 1 0 0 0 1 0 1]
# [B S B S S B B B S B S B S S B S S S B S B B S B B B S S S B B S S S B B B S B S]

def encode_to_nums(labels, s1, s2):
    string_to_num = {s1: -1, s2: 1}
    numerical_values = [string_to_num[string] for string in labels]
    return numerical_values


def revert_to_strings(labels, s1, s2):
    num_to_string = {-1: s1, 1: s2}
    reverted_strings = [num_to_string[num] for num in labels]
    return reverted_strings


def revert_to_string(label, s1, s2):
    num_to_string = {-1: s1, 1: s2}
    reverted_string = num_to_string[label]
    return reverted_string


def activation(y):
    return 1 if y >= 0 else -1


def perceptron_train(feat1Train, feat2train, T_class, learning, epoch):
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
    print(f'W = [{w0} , {w1} , {w2}]')
    w = [w0, w1, w2]
    return w


def perceptron_test(testsample1, testsample2, testclass, weight):
    y_predicted = []
    for i in range(len(testclass)):
        x1 = testsample1[i]
        x2 = testsample2[i]
        yk = testclass[i]
        y = x1 * weight[1] + x2 * weight[2] + weight[0]
        y = activation(y)
        print("Predicted", y, "Actual", yk)
        # print("Acutal", yk)
        y_predicted.append(y)
    return y_predicted


def perceptron_predict(x1, x2, W):
    return activation(x1 * W[1] + x2 * W[2] + W[0])


def PerceptronPlot(feature1, feature2, weights, labels):

    # Given weights and bias
    # w1, w2, b = 2, -3, 1

    # Sample data (replace these arrays with your feature arrays and class labels)
    feature1 = np.array(feature1)
    feature2 = np.array(feature2)
    classes = np.array(labels)  # Assuming binary classes 0 and 1

    # Define the slope and intercept of the decision boundary
    slope = -weights[1] / weights[2]
    # intercept = -weights[0] / weights[2]
    intercept = -weights[2] / weights[1]

    # Generate x1 values
    x1_values = np.linspace(min(feature1) - 1, max(feature2) + 1, 400)

    # Calculate corresponding x2 values using the decision boundary equation
    x2_values = slope * x1_values + intercept

    # Plot the data points
    plt.figure(figsize=(8, 6))
    # for f1,f2 in zip(feature1,feature2):
    #     if(f1)
    plt.scatter(feature1[classes == -1], feature2[classes == -1], color='b', label='Class 0')
    plt.scatter(feature1[classes == 1], feature2[classes == 1], color='r', label='Class 1')

    # Plot the decision boundary
    plt.plot(x1_values, x2_values, color='g', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.axhline(0, color='black', linewidth=0.5)  # X-axis
    plt.axvline(0, color='black', linewidth=0.5)  # Y-axis
    plt.grid(True, linewidth=0.2, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Decision Boundary for Single Perceptron Model')
    plt.show()


# 0.19999999999999996 -1.0192994731258205 -1.0819145688874579
def execute(chunk, feat1, feat2, lR, epch):
    reading = readFile(chunk)
    twoFeatures, targetClass = get_feature(feat1, feat2, reading)
    train1, train2, test1, test2, trainClassSample, testClassSample = train_test(twoFeatures, targetClass, feat1,
                                                                                 feat2)
    proF1train, proF2train, prof1test, prof2test, ClassTrain, ClassTest = preprocessing(train1, train2, test1, test2,
                                                                                        trainClassSample,
                                                                                        testClassSample)
    weights = perceptron_train(proF1train, proF2train, ClassTrain, lR, epch)
    PerceptronPlot(proF1train, proF2train, weights, ClassTrain)
    print("classTrain", ClassTrain)
    print("proF1train", proF1train)
    y_predicted = perceptron_test(prof1test, prof2test, ClassTest, weights)


execute(3, 'Area', 'Perimeter', 0.5, 100)
