import pandas as pd

df1 = pd.read_csv('Original_Arabic_Reviews.csv')
df2 = pd.read_csv('Test_Dataset_NoLabel.csv')

merged_df = df2.merge(df1[['review_description', 'rating']], on='review_description', how='left')
merged_df.to_csv('merged1.csv', encoding='utf-8-sig')
merged_df = pd.read_csv('merged1.csv')
merged_df.drop_duplicates(subset='ID', inplace=True)
merged_df.to_csv('TestLabeled.csv', encoding='utf-8-sig')

x1 = df2['review_description']
x2 = merged_df['review_description']

count = 0
for row in x1:
    if row in x2.values:
        count = count + 1

print(merged_df)

print('-'*100)

print(count)
print(count/10)

print('-'*100)

print(x1)
print(type(x1))

print('-'*100)

print(x2)
print(type(x2))

print('-'*100)

print(x1[0])
print(x2[0])

print('-'*100)

print(x1[0] in x2.values)
