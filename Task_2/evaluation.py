import matplotlib.pyplot as plt
import numpy as np

import Preprocessing


class Evaluation:
    def __init__(self,y_actual, y_predict):
        y_actual = np.array(y_actual)
        y_predict = np.array(y_predict)

        actual_labels = np.argmax(y_actual, axis=1)
        predicted_labels = np.argmax(y_predict, axis=1)

        accuracy = np.mean(actual_labels == predicted_labels)

        print(f'Accuracy: {accuracy * 100:.2f}%')



