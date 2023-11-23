# from Kernel import allY, weights, layers, feed_forward


ws = [
    [
        [0.21, 0.15],
        [-0.4, 0.1]
    ],
    [
        [-0.2, 0.3]
    ]
]

ys = [
    [0.43, 0.56],
    [0.42]
]


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
            for j, w in enumerate(weights[layer+1]):
                summation += w[i] * sigmas[0][j]
            current_sigma.append(y * (1 - y) * summation)
        sigmas.insert(0, current_sigma)
    return sigmas

# feed_forward()
# print(allY[-1])
error_signal = back_propagation(ys, [0], ws)
print(error_signal)
