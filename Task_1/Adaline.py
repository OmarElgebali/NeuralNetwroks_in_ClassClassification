import numpy as np
import numpy.random


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
        self.y_hat_test = []
        self.Xs = self.calcXs(feature1_train, feature2_train)
        self.calcWeights()

    def calcXs(self, x1, x2):
        Xs = []
        if self.isBias:
            for i in range(len(x1)):
                Xs.append([1, float(x1[i]), float(x2[i])])
            return Xs
        else:
            for i in range(len(x2)):
                Xs.append([float(x1[i]), float(x2[i])])
            return Xs

    def calcWeights(self):
        self.weights = [0.1 * np.random.rand() for _ in range(3)] if self.isBias else [0.1 * np.random.rand() for _ in range(2)]

    def fit(self):
        while self.mse >= self.mse_threshold and self.epochs > 0:
            for i in range(len(self.labels)):
                self.epochs -= 1
                self.y_hats[i] = np.dot(np.transpose(self.weights), self.Xs[i])
                self.y_hats[i] = 1 if self.y_hats[i] > 0 else 0
                self.errors[i] = self.labels[i] - self.y_hats[i]
                self.weights = [w + (self.eta * self.errors[i] * x) for w, x in zip(self.weights, self.Xs[i])]
            self.mse = sum([pow(e, 2) for e in self.errors]) / (2 * len(self.labels))

    def predict(self, inputXs):
        x_test_padded_with_one = [1, float(inputXs[0]), float(inputXs[1])] if self.isBias else [float(inputXs[0]), float(inputXs[1])]
        y_hat_test = np.dot(np.transpose(self.weights), x_test_padded_with_one)
        return 1 if y_hat_test > 0 else 0

    def test(self, feature1_test, feature2_test):
        y_hat_test = []
        Xs = self.calcXs(feature1_test, feature2_test)
        for x in Xs:
            dot_product = np.dot(np.transpose(self.weights), x)
            y_hat_test.append(1 if dot_product > 0 else 0)
        self.y_hat_test = y_hat_test
        return y_hat_test

    def test_with_eval(self, feature1_test, feature2_test, y_test):
        y_hat_test = []
        print(f"(Ya,Yp)")
        hit = 0
        Xs = self.calcXs(feature1_test, feature2_test)
        for i, x in enumerate(Xs):
            dot_product = np.dot(np.transpose(self.weights), x)
            # print(f"({y_test[i]} , {dot_product})")
            y_hat_test.append(1 if dot_product > 0 else 0)
            hit = (hit + 1) if y_test[i] == y_hat_test[-1] else hit
        print("< Adaline Test evaluation >")
        print(f"Total: {len(y_test)}")
        print(f"Hit  : {hit} = {(100 * hit)/len(y_test)}%")
        print(f"Miss : {len(y_test)-hit} = {(100 * (len(y_test)-hit))/len(y_test)}%")

        self.y_hat_test = y_hat_test
        return y_hat_test
