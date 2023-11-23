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
mean=0

def split_and_class(croppedData):
    data = croppedData.iloc[:, :5]
    label = croppedData['Class']
    return data, label


def train_test(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, stratify=label, random_state=22)
    return X_train, X_test, y_train, y_test


def FeatureNormalizingFit(algo, feature):
    scaler = MinMaxScaler()
    if algo == 'Perceptron':
        scaler = MinMaxScaler(feature_range=(-1, 1))
    normedFeature = np.array(feature).reshape(-1, 1)
    scaler.fit(normedFeature)
    return scaler


def FeatureNormlizeTransform(fitted, feature):
    normedFeatures = np.array(feature).reshape(-1, 1)
    transform_feature = fitted.transform(normedFeatures)
    return transform_feature


def EncoderFitter(target_class_train):
    global label_encode_model
    label_encode_model = LabelEncoder()
    label_encode_model.fit(target_class_train)
    return label_encode_model


def encoder_transform(algo, fitModel, target_class_array):
    Transformed_class = fitModel.transform(target_class_array.to_numpy())
    if algo == 'Perceptron':
        Transformed_class = [i if i != 0 else -1 for i in Transformed_class]
    return Transformed_class


def encoder_inverse_transform(algo, target_class_point):
    if algo == 'Perceptron':
        target_class_point = target_class_point if target_class_point == 1 else 0
    Transformed_class = label_encode_model.inverse_transform([target_class_point])[0]
    return Transformed_class


def fillEmpty(feature):
    features = feature.fillna(feature.mean())
    return features


def preprocessing_training_old(algo, feature_1_train, feature_2_train, trainClass):
    fillF1 = fillEmpty(feature_1_train)
    fillF2 = fillEmpty(feature_2_train)
    encoded_model = EncoderFitter(trainClass)
    encoded_target = encoder_transform(algo, encoded_model, trainClass)
    norm_model_f1 = FeatureNormalizingFit(algo, fillF1)
    norm_model_f2 = FeatureNormalizingFit(algo, fillF2)
    normed_feature_1_train = FeatureNormlizeTransform(norm_model_f1, fillF1)
    normed_feature_2_train = FeatureNormlizeTransform(norm_model_f2, fillF2)
    return encoded_model, norm_model_f1, norm_model_f2, encoded_target, normed_feature_1_train, normed_feature_2_train


def preprocessing_test_old(algo, testf1, testf2, testclass, norm1, norm2, encoder):
    classEncode = encoder_transform(algo, encoder, testclass)
    fillF1 = fillEmpty(testf1)
    fillF2 = fillEmpty(testf2)
    f1Transform = FeatureNormlizeTransform(norm1, fillF1)
    f2Transform = FeatureNormlizeTransform(norm2, fillF2)
    return classEncode, f1Transform, f2Transform


def prepare():
    dataset = pd.read_csv('Dry_Bean_Dataset.csv')
    data, target_class = split_and_class(dataset)
    x_train, x_test, y_train, y_test = train_test_split(data, target_class, test_size=0.3, stratify=label, random_state=22)


def preprocessing_training_new(x, y):
        mean = features = feature.fillna(feature.mean())