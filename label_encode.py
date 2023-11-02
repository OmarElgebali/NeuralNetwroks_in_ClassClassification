import pandas as pd

csv_file_path = 'cropped_file_P.csv'

df = pd.read_csv(csv_file_path)

# Extract the first two columns and the last column
selected_columns = df.iloc[:, [0, 1, -1]]
#------------------------------/---------------------------------------------
# Sample data with two strings in a column
# data = selected_columns[2]
data = selected_columns.iloc[:, -1]

print(data)



def encode_to_nums(l,s1,s2):
    string_to_num = {s1: -1, s2: 1}
    numerical_values = [string_to_num[string] for string in l]
    return numerical_values


def revert_to_strings(l ,s1,s2):
    num_to_string = {-1: s1, 1: s2}
    reverted_strings = [num_to_string[num] for num in l]
    return reverted_strings




numberd_label = encode_to_nums(data,"BOMBAY","CALI")
print(encode_to_nums(data,"BOMBAY","CALI"))
print(revert_to_strings(numberd_label,"BOMBAY","CALI"))




