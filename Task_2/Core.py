import pandas as pd
import numpy as np
import Kernel
import Preprocessing
from Task_2 import evaluation, confustion_matrix

x_train = y_train = x_test = y_test = []
model_activation_function = ""


# model_weights = []
# model_acc = 0.0
# model_MSE = 100


def preprocessing(activation_function, is_bias):
    global x_train, y_train, x_test, y_test, model_activation_function
    model_activation_function = activation_function
    x_train, y_train, x_test, y_test = Preprocessing.prepare(activation_function, is_bias)


# def convert_to_binary_target(prob_list):
#     maxProbability = (np.max(prob_list))
#     outputList = []
#     for yHat in prob_list:
#         if yHat == maxProbability:
#             outputList.append(1)
#         else:
#             outputList.append(0)
#     return outputList


def convert_to_binary_target_inline(prob_list):
    maxProbability = (np.max(prob_list))
    return [1 if prob == maxProbability else (0 if model_activation_function == 'Sigmoid' else -1) for prob in
            prob_list]


def predict(xs):
    ys_predicts, _ = Kernel.feed_forward(xs, model_activation_function)
    # Kernel.print_list_of_lists(ys_predicts, "Predict")
    y_predict = ys_predicts[-1]
    outputList = convert_to_binary_target_inline(y_predict)
    return outputList


# def fit(epochs, eta, bias, layers, neurons_list, progress_callback):
#     global model_acc, model_weights, model_MSE
#     model_acc = 0
#     model_MSE = 0
#     model_weights = []
#     y_predict = []
#     for curr_uw in range(1):
#         Kernel.generateWeights(neurons_list)
#         y_predict_t = []
#         mse = 0
#         updatedWeights = []
#         for epoch in range(epochs):
#             y_predict_t = []
#             mse = 0
#             for xs, ys_act in zip(x_train, y_train):
#                 ys, weights = Kernel.feed_forward(xs, model_activation_function)
#                 error_signal, error_sum = Kernel.back_propagation(ys, ys_act, weights, model_activation_function)
#                 mse += error_sum
#                 y_predict_t.append(convert_to_binary_target_inline(ys[-1]))
#                 network_xs = ys[:(len(ys) - 1)]
#                 for i in range(len(network_xs)):
#                     network_xs[i].insert(0, bias)
#                 network_xs.insert(0, xs)
#                 # print_list_of_lists(network_xs, 'network_xs')
#                 updatedWeights = Kernel.updateWeights(error_signal, eta, network_xs)
#             mse = mse / len(x_train)
#             progress_callback(epoch + 1, epochs)  # Emit progress update
#         print(f'MSE-{curr_uw}: {mse}')
#         acc = evaluation.evaluation_acc(y_train, y_predict_t)
#         # conf_mat = confustion_matrix.ConfusionMatrix(y_train, y_predict_t)
#         if acc >= model_acc and mse <= model_MSE:
#             model_weights = updatedWeights
#             model_acc = acc
#             model_MSE = mse
#     print(f'Final MSE: {model_MSE}')
#     print(f'Final Acc: {model_acc}')
#     print("=" * 500)
#     for xt in x_test:
#         y_predict.append(predict(xt))
#     _ = evaluation.evaluation_acc(y_test, y_predict)
#     _ = confustion_matrix.ConfusionMatrix(y_test, y_predict)


def fit(epochs, eta, bias, layers, neurons_list, progress_callback):
    y_predict = []
    y_predict_t = []
    mse = 0
    for epoch in range(epochs):
        mse = 0
        Kernel.generateWeights(neurons_list)
        for xs, ys_act in zip(x_train, y_train):
            ys, weights = Kernel.feed_forward(xs, model_activation_function)
            error_signal, error_sum = Kernel.back_propagation(ys, ys_act, weights, model_activation_function)
            mse += error_sum
            y_predict_t.append(convert_to_binary_target_inline(ys[-1]))
            network_xs = ys[:(len(ys) - 1)]
            for i in range(len(network_xs)):
                network_xs[i].insert(0, bias)
            network_xs.insert(0, xs)
            # print_list_of_lists(network_xs, 'network_xs')
            updatedWeights = Kernel.updateWeights(error_signal, eta, network_xs)
        mse = mse / len(x_train)
        progress_callback(epoch + 1, epochs)  # Emit progress update
    print(f'MSE: {mse}')
    _ = evaluation.evaluation_acc(y_train, y_predict_t)
    _ = confustion_matrix.ConfusionMatrix(y_train, y_predict_t)
    print("=" * 500)
    for xt in x_test:
        y_predict.append(predict(xt))
    _ = evaluation.evaluation_acc(y_test, y_predict)
    _ = confustion_matrix.ConfusionMatrix(y_test, y_predict)


def classify(xs):
    xs_preprocessed = Preprocessing.preprocessing_classification(xs)[0]
    return predict(xs_preprocessed)
