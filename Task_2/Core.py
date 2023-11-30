import numpy as np
import Kernel
import Preprocessing
from Kernel import feed_forward, back_propagation, generateWeights, print_list_of_lists, updateWeights
import evaluation
import confustion_matrix

x_train = y_train = x_test = y_test = []


def preprocessing(activation_function, is_bias):
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = Preprocessing.prepare(activation_function, is_bias)


finalWeights = []


def predict(xs):
    y = feed_forward(xs, "Sigmoid")[0][-1]
    maxProbability = (np.max(y))
    outputList = []
    for yHat in y:
        if yHat == maxProbability:
            outputList.append(1)
        else:
            outputList.append(0)
    return outputList


def fit(activation_function, epochs, eta, bias, layers, neurons_list):
    generateWeights(neurons_list)
    global finalWeights
    for _ in range(epochs):
        for xs, ys_act in zip(x_train, y_train):
            ys, weights = feed_forward(xs, activation_function)
            error_signal = back_propagation(ys, ys_act, weights)

            network_xs = ys[:(len(ys) - 1)]
            for i in range(len(network_xs)):
                network_xs[i].insert(0, bias)
            network_xs.insert(0, xs)
            # print_list_of_lists(network_xs, 'network_xs')
            updatedWeights = updateWeights(error_signal, eta, network_xs)
            finalWeights = updatedWeights
    for xt in x_test:
        y_predict.append(predict(xt))


y_predict = []
preprocessing('h', 1)
fit('h', 5000, 0.001, 1, 1,
    [5])
print(y_predict)
print(y_test)
accu = evaluation.Evaluation(y_test, y_predict)
confused = confustion_matrix.ConfusionMatrix(y_test, y_predict)


