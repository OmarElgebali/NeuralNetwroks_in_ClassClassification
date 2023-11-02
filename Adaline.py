import numpy as np
import numpy.random
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler


class Adaline:
    def __init__(self, feature1_train, feature2_train, labels, isBias, epochs, eta, mse_threshold):
        self.isBias = isBias
        self.epochs = epochs
        self.eta = eta
        self.labels = labels
        self.feature1_train = feature1_train
        self.feature2_train = feature2_train
        self.mse_threshold = mse_threshold
        self.mse = mse_threshold + 1
        self.weights = []
        self.y_hats = np.zeros(len(labels))
        self.errors = np.zeros(len(labels))
        self.y_actual = []
        self.Xs = self.calcXs(feature1_train, feature2_train)
        self.calcWeights()

    def calcXs(self, x1, x2):
        Xs = []
        if self.isBias:
            for i in range(len(x1)):
                Xs.append([1, x1[i], x2[i]])
            return Xs
        else:
            for i in range(len(x2)):
                Xs.append([x1[i], x2[i]])
            return Xs

    def calcWeights(self):
        if self.isBias:
            self.weights = [0.1 * np.random.rand() for _ in range(3)]
        else:
            self.weights = [0.1 * np.random.rand() for _ in range(2)]

    def fit(self):
        print("errors before while", self.errors)
        print("weights", self.weights)
        print("mse", self.mse)
        while self.mse >= self.mse_threshold and self.epochs > 0:
            for i in range(len(self.labels)):
                self.epochs -= 1
                print("epoch:", self.epochs)
                self.y_hats[i] = np.dot(np.transpose(self.weights), self.Xs[i])
                self.errors[i] = self.labels[i] - self.y_hats[i]
                self.weights = [w + (self.eta * self.errors[i] * x) for w, x in zip(self.weights, self.Xs[i])]
            self.mse = sum([pow(e, 2) for e in self.errors]) / (2 * len(self.labels))
        print("yhat", self.y_hats)
        print("errors", self.errors)
        print("weights", self.weights)
        print("mse", self.mse)

    def predict(self, inputXs):
        if self.isBias:
            x_test_padded_with_one = [1, inputXs[0], inputXs[1]]
        else:
            x_test_padded_with_one = [inputXs[0], inputXs[1]]

        y_hat_test = np.dot(np.transpose(self.weights), x_test_padded_with_one)
        if y_hat_test > 0:
            return 1
        else:
            return 0

    def test(self, feature1_test, feature2_test, y_actual):
        y_hat_test = []
        Xs = self.calcXs(feature1_test, feature2_test)
        for x in Xs:
            dot_product = np.dot(np.transpose(self.weights), x)
            if dot_product > 0:
                y_hat_test.append(1)
            else:
                y_hat_test.append(0)

        self.y_actual = y_actual
        return y_hat_test


#############
# tests
#############

dataset = read_csv('Dry_Bean_Dataset_ADALINE.csv')
x_1 = dataset['MajorAxisLength'].to_numpy().reshape(-1, 1)
x_2 = dataset['MinorAxisLength'].to_numpy().reshape(-1, 1)
y = dataset['Class'].tolist()
y_bin = [1 if label == 'CALI' else 0 for label in y]

minMax = [MinMaxScaler() for _ in range(2)]
x_1_scaled = minMax[0].fit_transform(x_1)
x_2_scaled = minMax[1].fit_transform(x_2)
adaline = Adaline(x_1_scaled, x_2_scaled, y_bin, 1, 1000, .1, .01)
adaline.fit()

feature1 = [295.4698306,
            274.8633573,
            313.570417,
            301.392791,
            357.1890036,
            330.155474,
            307.9956112,
            336.9594211,
            354.7254337,
            342.74242472,
            255.0735621,
            244.102719,
            233.8049677,
            240.9695005,
            260.0898266,
            257.4648986,
            265.4699791,
            252.7723761,
            257.4891433,
            240.3770452
            ]

feature2 = [
    196.3118225,
    211.8851392,
    197.8987002,
    208.3464437,
    179.8346914,
    198.4615253,
    213.6247263,
    199.0051127,
    193.354325,
    203.259513,
    157.80274,
    168.5224911,
    179.500919,
    175.6450402,
    163.1027196,
    165.036857,
    161.0284951,
    169.4350399,
    166.5592919,
    178.6807736
]
labelTest = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

feature1 = np.array(feature1).reshape(-1, 1)
feature2 = np.array(feature2).reshape(-1, 1)
feature1_scaled = minMax[0].transform(feature1)
feature2_scaled = minMax[1].transform(feature2)
sample1 = [240.3770452]
sample1 = np.array(sample1).reshape(-1, 1)

sample2 = [178.6807736]
sample2 = np.array(sample2).reshape(-1, 1)

f1 = minMax[0].transform(sample1)
f2 = minMax[1].transform(sample2)
predict = adaline.predict([f1, f2])
print("predict", predict)

adaline.test(feature1_scaled, feature2_scaled, labelTest)
