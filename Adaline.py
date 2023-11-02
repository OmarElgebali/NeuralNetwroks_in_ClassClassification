import numpy as np
import numpy.random
from pandas import read_csv


class Adaline:
    def __init__(self, feature1, feature2, labels, isBias, epochs, eta, mse_threshold):
        self.isBias = isBias
        self.epochs = epochs
        self.eta = eta
        self.labels = labels
        self.feature1 = feature1
        self.feature2 = feature2
        self.mse_threshold = mse_threshold
        self.Xs = []
        self.weights = []
        self.y_hats = np.zeros(len(labels))
        self.errors = np.zeros(len(labels))

    def calcXs(self):
        if self.isBias:
            for i in range(len(self.labels)):
                self.Xs.append(([1, self.feature1[i], self.feature2[i]]))
        else:
            for i in range(len(self.labels)):
                self.Xs.append(([self.feature1[i], self.feature2[i]]))

    def calcWeights(self):
        if self.isBias:
            self.weights = [0.1 * np.random.rand() for _ in range(3)]
        else:
            self.weights = [0.1 * np.random.rand() for _ in range(2)]

    def fit(self):
        for i in range(len(self.labels)):
            self.y_hats[i] = (np.dot(np.transpose(self.weights), self.Xs[i]))
            self.errors[i]=(self.y_hats[i] - self.labels[i])
            self.weights= self.weights + (self.eta*self.errors[i]*self.Xs[i])




dataset = read_csv('Dry_Bean_Dataset_ADALINE.csv')
print(dataset)

"""
Xs = 
[



]
"""
