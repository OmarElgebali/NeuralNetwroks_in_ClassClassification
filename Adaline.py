from pandas import read_csv


class Adaline:
    def __init__(self, feature_1, feature_2, labels, isBias, epochs, eta, mse_threshold):
        self.isBias = isBias
        self.epochs = epochs
        self.eta = eta
        self.mse_threshold = mse_threshold
        self.Xs = []
        for i in range(len(labels)):
            self.Xs.append([1, feature_1[i], feature_2[i]])
        self.weights = [ for ]


dataset = read_csv('Dry_Bean_Dataset_ADALINE.csv')
print(dataset)

"""
Xs = 
[



]
"""
