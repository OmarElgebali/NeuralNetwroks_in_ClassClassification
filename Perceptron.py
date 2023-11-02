import pandas as pd

csv_file_path = 'cropped_file_P.csv'

df = pd.read_csv(csv_file_path)

# Extract the first two columns and the last column
selected_columns = df.iloc[:, [0, 1, -1]]

# Save the selected columns to a new CSV file
output_csv_file_path = 'selected_columns.csv'
selected_columns.to_csv(output_csv_file_path, index=False)


