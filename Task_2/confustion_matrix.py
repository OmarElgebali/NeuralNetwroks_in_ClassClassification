import numpy as np
import matplotlib.pyplot as plt


class ConfusionMatrix:

    def __init__(self, actual_classes, predicted_classes, activation_function, case):
        actual_classes = np.array(actual_classes)
        predicted_classes = np.array(predicted_classes)

        num_classes = len(actual_classes[0])
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for actual, predicted in zip(actual_classes, predicted_classes):
            actual_index = np.argmax(actual)
            predicted_index = np.argmax(predicted)
            confusion_matrix[actual_index, predicted_index] += 1

        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'MLP\'s {case} Confusion Matrix with Act-Func: {activation_function}')
        plt.colorbar()

        classes = ['Class 0', 'Class 1', 'Class 2']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center')

        # plt.savefig(f'conf_matrix_{activation_function}_{case}.png')
        plt.show()
