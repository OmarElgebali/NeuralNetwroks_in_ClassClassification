from random import random
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hyper_tangent(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))


def label_lists(list_of_lists):
    num_of_lists = len(list_of_lists)
    labeled_lists = {}
    for i in range(num_of_lists):
        label = f'Layer {i + 1}'
        labeled_lists[label] = list_of_lists[i]

    return num_of_lists, labeled_lists


allY = []
# weights = []
# w1 = [[0.1, 0.3], [0.2, 0.4]]
# w2 = [[0.5, 0.7], [0.6, 0.8]]
# w3 = [[0.9, 0.2], [0.1, 0.3]]
# weights.append(w1)
# weights.append(w2)
# weights.append(w3)
# generated_weights = [
#     [
#         [-0.3, 0.21, 0.15],
#         [0.25, -0.4, 0.1]
#     ],
#     [
#         [-0.4, -0.2, 0.3]
#     ]
# ]


layers = 0
generated_weights = []


def Forward1(input, weights, layerNum, act_func):
    if layerNum == 0:
        return

    index = layers - layerNum
    neurons = []
    for r in weights[index]:
        a = 0
        print(f"Input: {input}")
        for w, x in zip(r, input):
            print(f"({w}, {x})")
            a += w * x

        sigma = sigmoid(a) if act_func == 'Sigmoid' else hyper_tangent(a)
        neurons.append(sigma)
        # print(activation)
    allY.append(neurons)
    Forward1(neurons, weights, layerNum - 1, act_func)


def feed_forward(inputs, act_func):
    Forward1(inputs, generated_weights, layers, act_func)
    num, labeled = label_lists(allY)

    print(f'Number of lists (Y): {num}')
    for label, lst in labeled.items():
        print(f'Y-{label}: {lst}')
    return allY, generated_weights


def back_propagation(outputs, actual, weights):
    sigmas = []
    sigma_y = []
    print(actual)
    # Output Layer
    for i, y in enumerate(outputs[-1]):
        print(f"ACT: {actual[i]} , Y: {y}")
        sigma_y.append((actual[i] - y) * y * (1 - y))
    sigmas.insert(0, sigma_y)

    # Hidden Layers
    for layer in reversed(range(len(outputs) - 1)):
        current_sigma = []
        for i, y in enumerate(outputs[layer]):
            summation = 0
            for j, w in enumerate(weights[layer + 1]):
                summation += w[i + 1] * sigmas[0][j]
            current_sigma.append(y * (1 - y) * summation)
        sigmas.insert(0, current_sigma)

    num, labeled = label_lists(sigmas)
    print(f'Number of lists (Sigma): {num}')
    for label, lst in labeled.items():
        print(f'Sigma - {label}: {lst}')
    return sigmas


def generateWeights(neurons_of_each_layer):
    global layers, generated_weights
    neurons_of_each_layer.append(3)
    layers = len(neurons_of_each_layer)
    input_size = 5
    AllWeights = []
    for number_neuron in neurons_of_each_layer:
        row_list = []
        for sublist in range(number_neuron):
            lst = [random() for _ in range(input_size)]
            rounded_lst = [round(num, 3) for num in lst]
            row_list.append(rounded_lst)
        input_size = number_neuron

        AllWeights.append(row_list)
    generated_weights = AllWeights
    num, labeled = label_lists(generated_weights)
    print(f'Number of lists (W): {num}')
    for label, lst in labeled.items():
        print(f'W - {label}: {lst}')
    # return AllWeights
