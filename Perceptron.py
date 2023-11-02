import pandas as pd

# Read the CSV file into a DataFrame
csv_file_path = 'Dry_Bean_Dataset.csv'
df = pd.read_csv(csv_file_path)

# Crop the first 101 rows
cropped_df = df.iloc[:100]

# Save the cropped DataFrame to a new CSV file
output_csv_file_path = 'cropped_file_P.csv'
cropped_df.to_csv(output_csv_file_path, index=False)
