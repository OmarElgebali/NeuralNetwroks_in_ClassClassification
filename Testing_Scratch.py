from pandas import read_csv

dataset = read_csv('Dry_Bean_Dataset_ADALINE.csv')
x_1 = dataset['MajorAxisLength'].tolist()
x_2 = dataset['MinorAxisLength'].tolist()
y = dataset['Class'].tolist()
y_bin = [1 if label == 'CALI' else 0 for label in y]

print(dataset)
print('=='*50)
print(x_1)
print('--'*50)
print(x_2)
print('--'*50)
print(y)
print('--'*50)
print(y_bin)
print('--'*50)
# print('--'*50)
# print(dataset['Class'])
