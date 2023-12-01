import random
import numpy as np


def label_lists(list_of_lists):
    num_of_lists = len(list_of_lists)
    labeled_lists = {}
    for i in range(num_of_lists):
        label = f'Layer {i + 1}'
        labeled_lists[label] = list_of_lists[i]
    return num_of_lists, labeled_lists


def print_list_of_lists(list_of_lists, its_name):
    num, labeled = label_lists(list_of_lists)
    print("="*400)
    print(f'Number of lists ({its_name}): {num}')
    for label, lst in labeled.items():
        print(f'{its_name} - {label}: {lst}')


def sigmoid(x):
    # print(x)
    return 1 / (1 + np.exp(-x))


def hyper_tangent(x):
    return np.tanh(x)


allY = []
layers = 0
generated_weights = []


def Forward1(input, weights, layerNum, act_func):
    if layerNum == 0:
        return

    index = layers - layerNum
    neurons = []
    for r in weights[index]:
        a = 0
        for w, x in zip(r, input):
            a += w * x

        sigma = sigmoid(a) if act_func == 'Sigmoid' else hyper_tangent(a)
        neurons.append(sigma)
    allY.append(neurons)
    Forward1(neurons, weights, layerNum - 1, act_func)


def feed_forward(inputs, act_func):
    global allY
    allY = []
    Forward1(inputs, generated_weights, layers, act_func)
    # print_list_of_lists(allY, 'Ys')
    return allY, generated_weights


def back_propagation(outputs, actual, weights, act_func):
    sigmas = []
    sigma_y = []

    # Output Layer
    error_sum = 0
    for i, y in enumerate(outputs[-1]):
        error_sum += (actual[i] - y) * (actual[i] - y)
        current_sigma = ((actual[i] - y) * y * (1 - y)) if act_func == 'Sigmoid' else ((actual[i] - y) * (1 - y) * (1 + y))
        sigma_y.append(current_sigma)
    sigmas.insert(0, sigma_y)

    # Hidden Layers
    for layer in reversed(range(len(outputs) - 1)):
        current_sigma = []
        for i, y in enumerate(outputs[layer]):
            summation = 0
            for j, w in enumerate(weights[layer + 1]):
                summation += w[i] * sigmas[0][j]
            current_sigma_hidden = (y * (1 - y) * summation) if act_func == 'Sigmoid' else ((1 + y) * (1 - y) * summation)
            current_sigma.append(current_sigma_hidden)
        sigmas.insert(0, current_sigma)

    # print_list_of_lists(sigmas, 'Sigma')
    return sigmas, error_sum


def generateWeights(neurons_of_each_layer):
    global layers, generated_weights
    neurons_of_each_layer.append(3)
    layers = len(neurons_of_each_layer)
    input_size = 5
    AllWeights = []
    for number_neuron in neurons_of_each_layer:
        row_list = []
        for sublist in range(number_neuron):
            lst = [random.random() for _ in range(input_size + 1)]
            row_list.append(lst)
        input_size = number_neuron

        AllWeights.append(row_list)
    generated_weights = AllWeights
    # print_list_of_lists(generated_weights, 'Weights')


def updateWeights(errors, learningRate, Xs):
    for layer, error_list in enumerate(errors):
        for neuron, neuron_error in enumerate(error_list):
            for i in range(len(generated_weights[layer][neuron])):
                generated_weights[layer][neuron][i] = (generated_weights[layer][neuron][i]
                                                       + learningRate * neuron_error * Xs[layer][i])
    # print_list_of_lists(generated_weights, 'Weights')
    return generated_weights

