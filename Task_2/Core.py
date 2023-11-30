import Preprocessing
from Kernel import feed_forward, back_propagation, generateWeights, generated_weights

x_train = y_train = x_test = y_test = []


def preprocessing(activation_function, is_bias):
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = Preprocessing.prepare(activation_function, is_bias)


def fit(activation_function, epochs, eta, bias, layers, neurons_list):
    # for _ in epochs:
    generateWeights(neurons_list)
    ys, ws = feed_forward(x_train, activation_function)
    error_signal = back_propagation(ys, y_train, ws)

# def predict(xs):


# def plot_draw(algorithm):
