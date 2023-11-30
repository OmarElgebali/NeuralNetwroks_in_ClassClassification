import pandas as pd

class_vales = ['BOMBAY', 'CALI', 'SIRA']

is_bias = 1
dataset = pd.read_csv('Dry_Bean_Dataset.csv')
print(f"Bias = {is_bias}")
dataset.insert(0, 'Bias', is_bias)

print(dataset[0])
# def target_encoder_model(target_values):
#     return [[1 if value == target_value else 0 for value in class_vales] for target_value in target_values]
#
#
# def inverse_target_encoder(target_class_points):
#     return [class_vales[index.index(1)] for index in target_class_points]
#
# arr = target_encoder_model(
#     ['BOMBAY', 'CALI', 'SIRA', 'CALI', 'SIRA', 'BOMBAY']
# )

# print(arr)
# print(inverse_target_encoder(arr))

for column in dataset.columns:
    # Accessing each column
    print("Column:", column)
    # Accessing column values
    for value in dataset[column]:
        print(value)
    print("----------------------")
