import pandas as pd
import numpy as np
import Kernel
import Preprocessing
from Core import x_train, y_train, x_test, y_test, model_activation_function, predict
from Kernel import feed_forward, back_propagation, generateWeights, print_list_of_lists, updateWeights, \
    generated_weights
from Task_2.Core import preprocessing, fit

# Xs = [[]]
# Ws=[[[]]]
preprocessing("Sigmoid", 1)
fit(1, 0.01, 1, 2, [3, 2])
print("#" * 100)
print("finalWeights", generated_weights)

# feed_forward([1], "Sigmoid")


# finalWeights [[[array([0.2176539 , 0.23168626, 0.2402908 ]), array([0.46904382, 0.47222875, 0.4739987 ]), array([0.16726743, 0.17094659, 0.17301377]), array([0.76223057, 0.76651333, 0.76895988]), array([0.74482113, 0.74807356, 0.74991046])], [array([0.72155389, 0.72248398, 0.72496351]), array([0.70102149, 0.70119452, 0.70150733]), array([1.00673521, 1.00694518, 1.0073435 ]), array([0.72802217, 0.72827871, 0.72879731]), array([0.85995381, 0.8601347 , 0.86049202])], [array([1.03080117, 1.03112206, 1.03396375]), array([0.6502596 , 0.65041487, 0.6508707 ]), array([1.00959481, 1.00976357, 1.01031661]), array([0.02662365, 0.02680735, 0.02749474]), array([0.60421147, 0.60435542, 0.60484882])]], [[array([0.97944149, 0.91490643, 0.90575948]), array([0.8264776 , 0.78438572, 0.77887127]), array([1.13335018, 1.08123416, 1.0739289 ])], [array([0.8078464 , 0.95995629, 1.01321126]), array([0.34440664, 0.44842454, 0.48519329]), array([0.43523891, 0.55911929, 0.60253307])]], [[array([-1.55568381, -1.53721797,  1.45927666]), array([-1.10301237, -1.08052615,  1.43874964])], [array([-1.72772677,  1.27503201, -1.69734649]), array([-0.94567293,  1.57820606, -0.93096735])], [array([-1.70071472, -1.68533763,  1.27542582]), array([-0.9973989 , -0.97892989,  1.51310495])]]]


# def convert_to_binary_target_inline(prob_list):
#     maxProbability = (np.max(prob_list))
#     return [1 if prob == maxProbability else 0 for prob in prob_list]


# Load the dataset
dataset_path = 'Dry_Bean_Dataset.csv'  # Replace with the path to your dataset
df = pd.read_csv(dataset_path)

# Get a single row (for example, the first row)
row_index = 0
single_row = df.iloc[row_index:row_index + 1]  # Use slicing to keep it as a DataFrame

# Extract the label
label_column = 'Class'  # Replace 'label' with the actual column name of your label
label = single_row[label_column].iloc[0]

# Remove the label from the row
features = single_row.drop(label_column, axis=1)

# Print the results
print("featuresValues", features.values)

xInputToPredict = Preprocessing.preprocessing_classification(features)

print("xPredict", xInputToPredict)
Kernel.print_list_of_lists(generated_weights, "finalWeights")
predict(xInputToPredict[0])

# def plot_draw(algorithm):
