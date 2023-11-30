import pandas as pd
import numpy as np
import Kernel
import Preprocessing

x_train = y_train = x_test = y_test = []
model_activation_function = ""


def preprocessing(activation_function, is_bias):
    global x_train, y_train, x_test, y_test, model_activation_function
    model_activation_function = activation_function
    x_train, y_train, x_test, y_test = Preprocessing.prepare(activation_function, is_bias)


def fit(epochs, eta, bias, layers, neurons_list):
    Kernel.generateWeights(neurons_list)
    for _ in range(epochs):
        for xs, ys_act in zip(x_train, y_train):
            ys, weights = Kernel.feed_forward(xs, model_activation_function)
            error_signal = Kernel.back_propagation(ys, ys_act, weights)

            network_xs = ys[:(len(ys) - 1)]
            for i in range(len(network_xs)):
                network_xs[i].insert(0, bias)
            network_xs.insert(0, xs)
            # print_list_of_lists(network_xs, 'network_xs')
            updatedWeights = Kernel.updateWeights(error_signal, eta, network_xs)


def convert_to_binary_target(prob_list):
    maxProbability = (np.max(prob_list))
    outputList = []
    for yHat in prob_list:
        if yHat == maxProbability:
            outputList.append(1)
        else:
            outputList.append(0)
    return outputList


# def convert_to_binary_target_inline(prob_list):
#     maxProbability = (np.max(prob_list))
#     return [1 if prob == maxProbability else 0 for prob in prob_list]


def predict(xs):
    ys_predicts, _ = Kernel.feed_forward(xs, model_activation_function)
    Kernel.print_list_of_lists(ys_predicts, "Predict")
    y_predict = ys_predicts[0][-1]
    outputList = convert_to_binary_target(y_predict)
    print("outputList", outputList)
    labels = Preprocessing.inverse_target_encoder([outputList])[0]
    print("Target: ", labels)
