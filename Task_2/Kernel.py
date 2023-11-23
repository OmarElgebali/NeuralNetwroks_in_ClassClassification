import pandas as pd
import numpy as np
import matplotlib as pl
from Preprocessing import prepare
import random



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
i = [0.3, 0.5]
weights = []
w1 = [[0.1, 0.3], [0.2, 0.4]]
w2 = [[0.5, 0.7], [0.6, 0.8]]
w3 = [[0.9, 0.2], [0.1, 0.3]]
weights.append(w1)
weights.append(w2)
weights.append(w3)
print(f'all weights : {weights}')
print('-'*50)
layers = 3


def Forward1(input, weights, layerNum):
    if layerNum == 0:
        return

    index = 3 - layerNum
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


Forward1(i, weights, 3)
num, labeled = label_lists(allY)

print(f'Number of lists: {num}')
for label, lst in labeled.items():
    print(f'{label}: {lst}')


x_train_processed, y_train_processed, x_test_processed, y_test_processed = prepare('Sigmoid')

print(type(x_train_processed))
print(len(x_train_processed))
for row in x_train_processed:
    print(row)


def generateWeights(number_hidden, neurons_of_each_layer):
    input_size = 5
    AllWeights = []
    neurons_of_each_layer.append(3)
    for number_neuron in neurons_of_each_layer:
        row_list = []
        for sublist in range(number_neuron):
            lst = [random.random() for _ in range(input_size)]
            rounded_lst = [round(num, 3) for num in lst]
            row_list.append(rounded_lst)
        input_size = number_neuron

        AllWeights.append(row_list)
    return AllWeights

w = generateWeights(4,[2, 4, 3, 2])
print(w)

