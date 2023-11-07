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

import Preprocessing
from Preprocessing import readFile, get_feature, train_test


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
    # print(f'W = [{w0} , {w1} , {w2}]')
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
        # print("Predicted", y, "Actual", yk)
        y_predicted.append(y)
    return y_predicted


def perceptron_predict(x1, x2, W):
    return activation(x1 * W[1] + x2 * W[2] + W[0])
