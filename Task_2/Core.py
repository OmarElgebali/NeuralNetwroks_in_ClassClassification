import Preprocessing
from Kernel import feed_forward, back_propagation, generateWeights
y_train = x_test = y_test = []
x_train = [1, 0, 0]


def preprocessing(activation_function):
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = Preprocessing.prepare(activation_function)


def fit(activation_function, epochs, eta, bias, layers, neurons_list):
    generateWeights(neurons_list)
    ys, ws = feed_forward(x_train, layers)
    error_signal = back_propagation(ys, [0], ws)


# def predict(xs):


# def plot_draw(algorithm):
