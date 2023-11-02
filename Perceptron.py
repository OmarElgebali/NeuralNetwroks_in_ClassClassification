import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file into a DataFrame
file_path = 'path/to/your/csv_file.csv'
df = pd.read_csv(file_path)

# Split the data for the first class (first 50 rows)
class1_data = df.iloc[:50, :]

# Split the data for the second class (next 50 rows)
class2_data = df.iloc[50:, :]

# Split the data for each class into train (25 rows) and test (25 rows) sets
class1_train, class1_test = train_test_split(class1_data, test_size=0.5, random_state=42)
class2_train, class2_test = train_test_split(class2_data, test_size=0.5, random_state=42)

# Concatenate the balanced train and test sets for both classes
train_data = pd.concat([class1_train, class2_train])
test_data = pd.concat([class1_test, class2_test])

# Print the shapes of train and test sets for verification
print("Train Data Shape:", train_data.shape)
print("Test Data Shape:", test_data.shape)
