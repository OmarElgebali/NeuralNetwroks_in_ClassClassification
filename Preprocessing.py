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

def readFile(radio):
    df = pd.read_csv('Dry_Bean_Dataset.csv')
    if radio == 1:
        select_row = df.iloc[0:100].reset_index()
    if radio == 2:
        half1 = df.iloc[:50]
        half2 = df.iloc[100:]
        select_row = pd.concat([half1, half2]).reset_index()
    elif radio == 3:
        select_row = df.iloc[50:150].reset_index()
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


def EncoderFitter(Tclasstrain):
    global label_encode_model
    label_encode_model = LabelEncoder()
    label_encode_model.fit(Tclasstrain)
    return label_encode_model


def EncoderTansformed(algo, fitModel, tclass):
    Transformed_class = fitModel.transform(tclass.to_numpy())
    if algo == 'Perceptron':
        Transformed_class = [i if i != 0 else -1 for i in Transformed_class]
    return Transformed_class


def EncoderInvereseTansformed(algo, tclass):
    if algo == 'Perceptron':
        tclass = tclass if tclass == 1 else 0
    Transformed_class = label_encode_model.inverse_transform([tclass])[0]
    return Transformed_class


def fillEmpty(feature):
    features = feature.fillna(feature.mean())
    return features


def preprocessing_training(algo, trainf1, trainf2, trainClass):
    fillF1 = fillEmpty(trainf1)
    fillF2 = fillEmpty(trainf2)
    EncodedModel = EncoderFitter(trainClass)
    encodedTarget = EncoderTansformed(algo, EncodedModel, trainClass)
    normModelf1 = FeatureNormalizingFit(algo, fillF1)
    normModelf2 = FeatureNormalizingFit(algo, fillF2)
    normedFeaturetrain1 = FeatureNormlizeTransform(normModelf1, fillF1)
    normedFeaturetrain2 = FeatureNormlizeTransform(normModelf2, fillF2)
    return EncodedModel, normModelf1, normModelf2, encodedTarget, normedFeaturetrain1, normedFeaturetrain2


def preprocessing_test(algo, testf1, testf2, testclass, norm1, norm2, encoder):
    classEncode = EncoderTansformed(algo, encoder, testclass)
    fillF1 = fillEmpty(testf1)
    fillF2 = fillEmpty(testf2)
    f1Transform = FeatureNormlizeTransform(norm1, fillF1)
    f2Transform = FeatureNormlizeTransform(norm2, fillF2)
    return classEncode, f1Transform, f2Transform


# df = readFile(1)
# print("Data", df)
#
# feature, targetclass = get_feature('Area', 'Perimeter', df)
# print("Features", feature)
# print("Target Class", targetclass)
#
# f1train, f2train, f1test, f2test, Ctrain, Ctest = train_test(feature, targetclass, 'Area', 'Perimeter')
# print("Feature 1 train & 2 train", f1train, f2train)
# print("Feature 1 test & 2 test", f1test, f2test)
# print("Target Class train", Ctrain)
# print("Target Class test", Ctest)
#
# Encoder, normalizer1, normalizer2, encodedclasstrain, norfeattrain1, norFeattrain2 = preprocessing_training(f1train, f2train, Ctrain)
# print("Encoded Target Class Train ", encodedclasstrain)
# print("Normalized Feature Train 1 ", norfeattrain1)
# print("Normalized Feature Train 2 ", norFeattrain2)
#
# classEncodetest, f1Transformtest, f2Transformtest = preprocessing_test(f1test, f2test, Ctest, normalizer1, normalizer2, Encoder)
# print("Encoded Target Class Test ", classEncodetest)
# print("Transformed Feature 1 Test ", f1Transformtest)
# print("Transformed Feature 2 Test ", f2Transformtest)


# weight = perceptron_train(norfeattrain1, norFeattrain2, encodedclasstrain, 0.5, 100)
# print(weight)
# Bigteste(f1Transformtest, f2Transformtest, classEncodetest, weight)
