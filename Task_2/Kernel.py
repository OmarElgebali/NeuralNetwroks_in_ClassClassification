import pandas as pd

croppedData = pd.read_csv('Dry_Bean_Dataset.csv')
data = croppedData.iloc[:, :5]
label = croppedData['Class']

print(data)
print(label)
