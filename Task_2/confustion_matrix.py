import numpy as np
import matplotlib.pyplot as plt


# Step 2: Define actual and predicted classes in one-hot encoded format
# actual_classes = np.array(
#     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
# predicted_classes = np.array(
#     [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
#

class ConfusionMatrix:

    def __init__(self, actual_classes, predicted_classes):
        actual_classes = np.array(actual_classes)
        predicted_classes = np.array(predicted_classes)
        # Step 3: Initialize confusion matrix
        num_classes = len(actual_classes[0])
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Step 4: Populate confusion matrix
        for actual, predicted in zip(actual_classes, predicted_classes):
            actual_index = np.argmax(actual)
            predicted_index = np.argmax(predicted)
            confusion_matrix[actual_index, predicted_index] += 1

        # Step 5: Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
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

        # plt.show()
        plt.savefig('conf_matrix.png')

