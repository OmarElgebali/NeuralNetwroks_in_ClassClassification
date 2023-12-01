from random import random
import tkinter as tk
from tkinter import W
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler

mean = []
scaler_models = []
model_bias = -1
class_values = ['BOMBAY', 'CALI', 'SIRA']
model_act_func = ''


def split_and_class(croppedData):
    data = croppedData.iloc[:, :5]
    label = croppedData['Class']
    return data, label


def train_test(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, stratify=label, random_state=22)
    return X_train, X_test, y_train, y_test


def feature_normalizing_fit(dataset, activation_function):
    global scaler_models
    scaler_models = []
    for column in dataset.columns:
        scaler_obj = StandardScaler() if activation_function == 'Sigmoid' else StandardScaler()
        scaler_obj.fit(dataset[[column]])
        scaler_models.append(scaler_obj)


def feature_normalize_transform(dataset):
    for index, column in enumerate(dataset.columns):
        dataset[[column]] = scaler_models[index].transform(dataset[[column]])
    return dataset


def target_encoder_model(target_values):
    return [[1 if value == target_value else (0 if model_act_func == 'Sigmoid' else -1) for value in class_values] for target_value in target_values]
    # return [[1 if value == target_value else 0 for value in class_values] for target_value in target_values]


def inverse_target_encoder(target_class_points):
    return [class_values[index.index(1)] for index in target_class_points]


def fillEmptyTrain(dataset):
    global mean
    mean = dataset.mean()
    return dataset.fillna(mean)


def fillEmptyTest(dataset):
    return dataset.fillna(mean)


def preprocessing_training(x, y, activation_function):
    global mean, model_act_func
    mean = []
    model_act_func = activation_function
    x_filled = fillEmptyTrain(x)
    feature_normalizing_fit(x_filled, activation_function)
    normalized_x = feature_normalize_transform(x_filled)
    encoded_target = target_encoder_model(y)
    # encoder_fit(y)
    # encoded_target = encoder_transform(y)
    return normalized_x, encoded_target


def preprocessing_testing(x, y):
    x_filled = fillEmptyTest(x)
    normalized_x = feature_normalize_transform(x_filled)
    encoded_target = target_encoder_model(y)
    # encoded_target = encoder_transform(y)
    return normalized_x, encoded_target


def prepare(activation_function, is_bias):
    global model_bias
    dataset = pd.read_csv('Dry_Bean_Dataset.csv')
    data, target_class = split_and_class(dataset)
    x_train, x_test, y_train, y_test = train_test_split(data, target_class, test_size=0.3, stratify=target_class,
                                                        random_state=22)
    x_train_processed, y_train_processed = preprocessing_training(x_train, y_train, activation_function)
    x_test_processed, y_test_processed = preprocessing_testing(x_test, y_test)
    model_bias = is_bias
    x_train_processed.insert(0, 'Bias', is_bias)
    x_test_processed.insert(0, 'Bias', is_bias)
    x_train_listed = x_train_processed.values.tolist()
    x_test_listed = x_test_processed.values.tolist()
    return x_train_listed, y_train_processed, x_test_listed, y_test_processed


def preprocessing_classification(dataset):
    normalizedDataSet = feature_normalize_transform(dataset)
    normalizedDataSet.insert(0, 'Bias', model_bias)
    return normalizedDataSet.values.tolist()
