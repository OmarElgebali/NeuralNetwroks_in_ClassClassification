import matplotlib.pyplot as plt
import numpy as np

import Preprocessing


class Evaluation:
    def __init__(self, y_predict, y_actual, algorithm):
        self.y_actual = y_actual
        self.y_predict = y_predict
        self.algorithm = algorithm

    def confusion_matrix(self):
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for actual, predicted in zip(self.y_actual, self.y_predict):
            if actual == 1 and predicted == 1:
                true_positives += 1
            elif actual == 0 and predicted == 1:
                false_positives += 1
            elif actual == 0 and predicted == 0:
                true_negatives += 1
            elif actual == 1 and predicted == 0:
                false_negatives += 1
        confusion_matrix = [
            [true_positives, false_positives],
            [false_negatives, true_negatives]
        ]
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(f"Confusion Matrix of {self.algorithm} Algorithm")
        plt.ylabel("True labels")
        plt.xlabel("Predicted labels")
        plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        plt.show()
        # for row in confusion_matrix:
        #     print(row)

    def PerceptronPlot(self, feature1, feature2, weights, labels, f1_name, f2_name):
        min_class = 0 if self.algorithm == 'Adaline' else -1

        feature1 = np.array(feature1)
        feature2 = np.array(feature2)
        classes = np.array(labels)

        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]

        x1_values = np.linspace(min(feature1) - 1, max(feature2) + 1, 400)

        x2_values = slope * x1_values + intercept

        plt.figure(figsize=(8, 6))
        plt.scatter(feature1[classes == min_class], feature2[classes == min_class], color='b', label=Preprocessing.EncoderInvereseTansformed(self.algorithm, 0))
        plt.scatter(feature1[classes == 1], feature2[classes == 1], color='r', label=Preprocessing.EncoderInvereseTansformed(self.algorithm, 1))

        plt.plot(x1_values, x2_values, color='g', label='Decision Boundary')

        plt.xlabel(f'Feature 1 ({f1_name})')
        plt.ylabel(f'Feature 2 ({f2_name})')
        plt.axhline(0, color='black', linewidth=0.5)  # X-axis
        plt.axvline(0, color='black', linewidth=0.5)  # Y-axis
        plt.grid(True, linewidth=0.2, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title(f'Decision Boundary for {self.algorithm} Model')
        plt.show()
