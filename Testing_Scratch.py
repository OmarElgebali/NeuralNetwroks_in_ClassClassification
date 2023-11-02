from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

dataset = read_csv('Dry_Bean_Dataset_ADALINE.csv')
x_1 = dataset['MajorAxisLength'].to_numpy().reshape(-1, 1)
x_2 = dataset['MinorAxisLength'].to_numpy().reshape(-1, 1)
y = dataset['Class'].tolist()
y_bin = [1 if label == 'CALI' else 0 for label in y]

minMax = [MinMaxScaler() for _ in range(2)]
x_1_scaled = minMax[0].fit_transform(x_1)
x_2_scaled = minMax[1].fit_transform(x_2)

print(dataset)
print('=='*50)
print(x_1)
print('--'*50)
print(x_2)
print('--'*50)
print(x_1_scaled)
print('--'*50)
print(x_2_scaled)
print('--'*50)
# print(y)
# print('--'*50)
# print(y_bin)
# print('--'*50)
# print('--'*50)
# print(dataset['Class'])
# import numpy as np
#
# w = [0.4, -0.4, 0.4]
# x = [1.0, 1.0, -1.0]
# # r = w + (n * e) * x
# r = np.dot(np.transpose(w), x)
#
# print(r)

