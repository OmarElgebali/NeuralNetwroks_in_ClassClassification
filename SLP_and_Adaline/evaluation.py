import matplotlib.pyplot as plt
import numpy as np

import Preprocessing


class Evaluation:
    def __init__(self, y_predict, y_actual, algorithm, f1_name, f2_name):
        self.y_actual = y_actual
        self.y_predict = y_predict
        self.algorithm = algorithm
        self.f1_name = f1_name
        self.f2_name = f2_name
        self.class_0 = Preprocessing.encoder_inverse_transform(algorithm, 0)
        self.class_1 = Preprocessing.encoder_inverse_transform(algorithm, 1)

    def confusion_matrix(self):
        minimum_class = 0
        if self.algorithm == 'Perceptron':
            minimum_class = -1
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        for actual, predicted in zip(self.y_actual, self.y_predict):
            if actual == 1 and predicted == 1:
                true_positives += 1
            elif actual == minimum_class and predicted == 1:
                false_positives += 1
            elif actual == minimum_class and predicted == minimum_class:
                true_negatives += 1
            elif actual == 1 and predicted == minimum_class:
                false_negatives += 1
        confusion_matrix = [
            [true_positives, false_positives],
            [false_negatives, true_negatives]
        ]
        # accuracy = (true_negatives + true_positives) / len(self.y_actual)
        accuracy = ( (true_negatives + true_positives) * 100) / (true_negatives + true_positives + false_negatives + false_positives)
        plt.close('all')
        plt.figure(clear=True, figsize=(7, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.title(f"Confusion Matrix of {self.algorithm} Algorithm [Accuracy = {accuracy}%]\n with Feature 1 ({self.f1_name}) and Feature 2 ({self.f2_name}) \n for the 2 classes ({self.class_0} & {self.class_1})")
        plt.ylabel("True labels")
        plt.xlabel("Predicted labels")
        plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
        plt.yticks([0, 1], ["True 0", "True 1"])
        # plt.savefig(f"Results/[{self.algorithm}][{self.f1_name}_{self.f2_name}][{self.class_0}_{self.class_1}] - Confusion Matrix.png")
        plt.show()

    def plot_decision_boundary(self, feature1, feature2, weights, labels):
        min_class = 0 if self.algorithm == 'Adaline' else -1

        feature1 = np.array(feature1)
        feature2 = np.array(feature2)
        classes = np.array(labels)

        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]

        x1_values = np.linspace(min(feature1) - 1, max(feature2) + 1, 400)

        x2_values = slope * x1_values + intercept

        plt.close('all')
        plt.figure(clear=True, figsize=(8, 6))
        plt.scatter(feature1[classes == min_class], feature2[classes == min_class], color='b', label=self.class_0)
        plt.scatter(feature1[classes == 1], feature2[classes == 1], color='r', label=self.class_1)

        plt.plot(x1_values, x2_values, color='g', label='Decision Boundary')

        plt.xlabel(f'Feature 1 ({self.f1_name})')
        plt.ylabel(f'Feature 2 ({self.f2_name})')
        plt.axhline(0, color='black', linewidth=0.5)  # X-axis
        plt.axvline(0, color='black', linewidth=0.5)  # Y-axis
        plt.grid(True, linewidth=0.2, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title(f"Decision Boundary of {self.algorithm} Algorithm\n with Feature 1 ({self.f1_name}) and Feature 2 ({self.f2_name}) \n for the 2 classes ({self.class_0} & {self.class_1})")
        # plt.savefig(f"Results/[{self.algorithm}][{self.f1_name}_{self.f2_name}][{self.class_0}_{self.class_1}] - Decision Boundary.png")
        plt.show()
