# import pandas as pd
# import numpy as np
# import matplotlib as pl
# from Preprocessing import prepare
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def label_lists(list_of_lists):
#     num_of_lists = len(list_of_lists)
#     labeled_lists = {}
#     for i in range(num_of_lists):
#         label = f'Layer {i + 1}'
#         labeled_lists[label] = list_of_lists[i]
#
#     return num_of_lists, labeled_lists
#
#
# allY = []
# i = [0.3, 0.5]
# weights = []
# w1 = [[0.1, 0.3], [0.2, 0.4]]
# w2 = [[0.5, 0.7], [0.6, 0.8]]
# w3 = [[0.9, 0.2], [0.1, 0.3]]
# weights.append(w1)
# weights.append(w2)
# weights.append(w3)
# layers = 3
#
#
# def Forward1(input, weights, layerNum):
#     if layerNum == 0:
#         return
#
#     index = 3 - layerNum
#     neurons = []
#     for r in weights[index]:
#         a = 0.5
#         for w, x in zip(r, input):
#             a += w * x
#
#         sigma = sigmoid(a)
#         neurons.append(sigma)
#         # print(activation)
#     allY.append(neurons)
#     Forward1(neurons, weights, layerNum - 1)
#
#
# Forward1(i, weights, 3)
# num, labeled = label_lists(allY)
#
# print(f'Number of lists: {num}')
# for label, lst in labeled.items():
#     print(f'{label}: {lst}')
#
# x_train_processed, y_train_processed, x_test_processed, y_test_processed = prepare('Sigmoid')

# [[[neuron1 weights]  [neuron2 weights] [neuron3 weights]] ======> layer 1
# [[neuron1 weights]  [neuron2 weights] [neuron3 weights]] ======> layer 2
# [[neuron1 weights]  [neuron2 weights] [neuron3 weights]] ======> layer 3...etc

# [
# [  [w,w,w,w],        [[],[],[],[]],          [[],[],[],[]]
# [  [[],[],[],[]],        [[],[],[],[]],          [[],[],[],[]]
# [  [[],[],[],[]],        [[],[],[],[]],          [[],[],[],[]]
# ]
# weights = \
#     [
#         [[-.3, .21, .15], [.25, -.4, .1]],
#         [[-.4, -.2, .3]]
#     ]
#
# errors = [
#     [.005, -.007],
#     [-.102]
# ]
# Xs = \
#     [
#         [[1, 0, 0], [1, 0, 0]],
#         [[1, 0.43, 0.56]]
#     ]
#
# newWeights = []


# DON'T DELETE THIS!!!!  will use it later...
# def updateWeights(weights, errors, learningRate, Xs):
#     for layer, error_list in enumerate(errors):
#         for neuron, neuron_error in enumerate(error_list):
#             for i in range(len(weights[layer][neuron])):
#                 weights[layer][neuron][i] = weights[layer][neuron][i] + \
#                                             learningRate * neuron_error * Xs[layer][neuron][i]
#     return weights

weights = \
    [
        [[-.4, .2, .4, -.5], [.2, -.3, .1, .2]],
        [[.1, -.3, -.2]]
    ]

errors = [
    [-.0087, .0065],
    [.1311]
]
Xs = \
    [
        [1, 1, 0, 1]
        , [1, 0.332, 0.525]
    ]


def updateWeights(weights, errors, learningRate, Xs):
    for layer, error_list in enumerate(errors):
        for neuron, neuron_error in enumerate(error_list):
            for i in range(len(weights[layer][neuron])):
                weights[layer][neuron][i] = weights[layer][neuron][i] + \
                                            learningRate * neuron_error * Xs[layer][i]
    return weights


# learning rate=.01
print(updateWeights(weights, errors, 0.9, Xs))

# first output
# [[[-0.29995, 0.21, 0.15], [0.24993, -0.4, 0.1]], [[-0.40102000000000004, -0.20043860000000002, 0.2994288]]]

# second output
# [[[-0.40783, 0.19217, 0.4, -0.50783], [0.20585, -0.29414999999999997, 0.1, 0.20585]], [[0.21799000000000002, -0.26082732, -0.13805525000000002]]]