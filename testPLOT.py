import matplotlib.pyplot as plt
import numpy as np

# Given weights and bias
w1, w2, b = 2, -3, 1

# Sample data (replace these arrays with your feature arrays and class labels)
feature1 = np.array([2, 3, 1, 5, 6, 4, 7])
feature2 = np.array([3, 2, 5, 1, 7, 6, 4])
classes = np.array([0, 1, 0, 1, 1, 0, 1])  # Assuming binary classes 0 and 1

# Define the slope and intercept of the decision boundary
slope = -w1 / w2
intercept = -b / w2

# Generate x1 values
x1_values = np.linspace(min(feature1) - 1, max(feature1) + 1, 400)

# Calculate corresponding x2 values using the decision boundary equation
x2_values = slope * x1_values + intercept

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(feature1[classes == 0], feature2[classes == 0], color='b', label='Class 0')
plt.scatter(feature1[classes == 1], feature2[classes == 1], color='r', label='Class 1')

# Plot the decision boundary
plt.plot(x1_values, x2_values, color='g', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.axhline(0, color='black', linewidth=0.5)  # X-axis
plt.axvline(0, color='black', linewidth=0.5)  # Y-axis
plt.grid(True, linewidth=0.2, linestyle='--', alpha=0.7)
plt.legend()
plt.title('Decision Boundary for Single Perceptron Model')
plt.show()


#-------------------------------------------------------------------------
 # data = {
 #    'Column1': proF1train,
 #    'Column2': proF2train,
 #    'Column3': ClassTrain
 #    }
 #
 #    df = pd.DataFrame(data)
 #
 #    df.to_csv('output_to_plot.csv', index=False)


# import numpy as np
# import matplotlib.pyplot as plt
#
# # Example weights and bias from your perceptron model
# w1 = 0.5
# w2 = -0.3
# b = 0.1
#
# # Generate random data points for feature one and feature two
# feature_one = np.random.rand(100)  # Example data for feature one
# feature_two = np.random.rand(100)  # Example data for feature two
#
# # Calculate decision boundary
# x_values = np.linspace(-1, 1, 100)  # Adjust the range according to your data
# y_values = (-w1 * x_values - b) / w2
#
# # Plot data points
# plt.scatter(feature_one, feature_two, color='blue', label='Class 1')
# # Assuming feature_one on x-axis and feature_two on y-axis
#
# # Plot decision boundary
# plt.plot(x_values, y_values, color='red', label='Decision Boundary')
#
# plt.xlabel('Feature One')
# plt.ylabel('Feature Two')
# plt.legend()
# plt.show()
#--------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
#
# def PerceptronPlot(f1, f2, labels, weights):
#     # Normalize feature values to a smaller range for better visualization
#     min_val = min(min(f1), min(f2))
#     max_val = max(max(f1), max(f2))
#     f1_normalized = (f1 - min_val) / (max_val - min_val)
#     f2_normalized = (f2 - min_val) / (max_val - min_val)
#
#     x_values = np.linspace(0, 1, 100)  # Adjust the range according to your normalized data
#     y_values = (-weights[1] * x_values - weights[0]) / weights[2]
#
#     # Plot data points for each class with different colors
#     for label in np.unique(labels):
#         class_indices = np.where(labels == label)
#         print("class indices : ",class_indices)
#         plt.scatter(f1_normalized[class_indices], f2_normalized[class_indices], label=f'Class {label}')
#
#     # Plot decision boundary
#     plt.plot(x_values, y_values, color='red', label='Decision Boundary')
#
#     plt.xlabel('Feature One')
#     plt.ylabel('Feature Two')
#     plt.legend()
#     plt.show()
