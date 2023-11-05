import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocessing(algo, trainf1, trainf2, testf1, testf2, trainClass, testClass):
    le = LabelEncoder()
    if algo == 'Perceptron':
        scaler = MinMaxScaler(feature_range=(-1, 1))
        encodedTrainC = le.fit_transform(trainClass)
        encodedTrainC = [i if i != 0 else -1 for i in encodedTrainC]
        trainf1 = trainf1.fillna(trainf1.mean())
        trainf2 = trainf2.fillna(trainf2.mean())
        normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
        testf1 = testf1.fillna(testf1.mean())
        normF1_test = scaler.transform(np.array(testf1).reshape(-1, 1))
        encodedTestC = le.transform(testClass)
        encodedTestC = [i if i != 0 else -1 for i in encodedTestC]
        testf2 = testf2.fillna(testf2.mean())
        normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
        normF2_test = scaler.transform(np.array(testf2).reshape(-1, 1))

    else:
        scaler = MinMaxScaler()
        encodedTrainC = le.fit_transform(trainClass)
        trainf1 = trainf1.fillna(trainf1.mean())
        trainf2 = trainf2.fillna(trainf2.mean())
        normF1_train = scaler.fit_transform(np.array(trainf1).reshape(-1, 1))
        testf1 = testf1.fillna(testf1.mean())
        normF1_test = scaler.transform(np.array(testf1).reshape(-1, 1))
        normF2_train = scaler.fit_transform(np.array(trainf2).reshape(-1, 1))
        encodedTestC = le.transform(testClass)
        testf2 = testf2.fillna(testf2.mean())
        normF2_test = scaler.transform(np.array(testf2).reshape(-1, 1))

    return normF1_train, normF2_train, normF1_test, normF2_test, encodedTrainC, encodedTestC
