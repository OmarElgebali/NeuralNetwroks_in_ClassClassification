import pandas as pd
import numpy as np
import matplotlib as pl
from Preprocessing import prepare


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
layers = 0

weights = [
    [
        [-0.3, 0.21, 0.15],
        [0.25, -0.4, 0.1]
    ],
    [
        [-0.4, -0.2, 0.3]
    ]
]


def Forward1(input, weights, layerNum):
    if layerNum == 0:
        return

    index = layers - layerNum
    neurons = []
    for r in weights[index]:
        a = 0
        for w, x in zip(r, input):
            a += w * x

        sigma = sigmoid(a)
        neurons.append(sigma)
        # print(activation)
    allY.append(neurons)
    Forward1(neurons, weights, layerNum - 1)


def feed_forward(inputs, num_of_layers):
    global layers
    layers = num_of_layers
    Forward1(inputs, weights, layers)
    num, labeled = label_lists(allY)

    print(f'Number of lists (Y): {num}')
    for label, lst in labeled.items():
        print(f'Y-{label}: {lst}')
    return allY, weights


def back_propagation(outputs, actual, weights):
    sigmas = []
    sigma_y = []
    # Output Layer
    for i, y in enumerate(outputs[-1]):
        sigma_y.append((actual[i] - y) * y * (1 - y))
    sigmas.insert(0, sigma_y)
    # Hidden Layers
    for layer in reversed(range(len(outputs) - 1)):
        current_sigma = []
        for i, y in enumerate(outputs[layer]):
            summation = 0
            for j, w in enumerate(weights[layer + 1]):
                summation += w[i+1] * sigmas[0][j]
            current_sigma.append(y * (1 - y) * summation)
        sigmas.insert(0, current_sigma)

    num, labeled = label_lists(sigmas)
    print(f'Number of lists (Sigma): {num}')
    for label, lst in labeled.items():
        print(f'Sigma - {label}: {lst}')
    return sigmas
