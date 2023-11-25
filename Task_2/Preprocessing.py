from random import random
import tkinter as tk
from tkinter import W
from tkinter import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

label_encode_model = LabelEncoder()
mean = []
scaler_models = []


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
        scaler_obj = MinMaxScaler() if activation_function == 'Sigmoid' else MinMaxScaler(feature_range=(-1, 1))
        scaler_obj.fit(dataset[[column]])
        scaler_models.append(scaler_obj)


def feature_normalize_transform(dataset):
    for index, column in enumerate(dataset.columns):
        dataset[[column]] = scaler_models[index].transform(dataset[[column]])
    return dataset


def encoder_fit(target_class_train):
    global label_encode_model
    label_encode_model = LabelEncoder()
    label_encode_model.fit(target_class_train)


def encoder_transform(target_class_array):
    return label_encode_model.transform(target_class_array.to_numpy())


def encoder_inverse_transform(target_class_point):
    return label_encode_model.inverse_transform(target_class_point)


def fillEmptyTrain(dataset):
    global mean
    mean = dataset.mean()
    return dataset.fillna(mean)


def fillEmptyTest(dataset):
    return dataset.fillna(mean)


def preprocessing_training(x, y, activation_function):
    global mean
    mean = []
    x_filled = fillEmptyTrain(x)
    feature_normalizing_fit(x_filled, activation_function)
    normalized_x = feature_normalize_transform(x_filled)
    encoder_fit(y)
    encoded_target = encoder_transform(y)
    return normalized_x, encoded_target


def preprocessing_testing(x, y):
    x_filled = fillEmptyTest(x)
    normalized_x = feature_normalize_transform(x_filled)
    encoded_target = encoder_transform(y)
    return normalized_x, encoded_target


def prepare(activation_function):
    dataset = pd.read_csv('Dry_Bean_Dataset.csv')
    data, target_class = split_and_class(dataset)
    x_train, x_test, y_train, y_test = train_test_split(data, target_class, test_size=0.3, stratify=target_class,
                                                        random_state=22)
    x_train_processed, y_train_processed = preprocessing_training(x_train, y_train, activation_function)
    x_test_processed, y_test_processed = preprocessing_testing(x_test, y_test)
    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


def preprocessing_classification(dataset):
    return feature_normalize_transform(fillEmptyTest(dataset))
