import nltk
import pandas as pd

df = pd.read_csv('TestLabeled.csv')

# Assuming df is your DataFrame
# Count of each value in the 'Rating' column
rating_counts = df['rating'].value_counts()

# Display counts and percentages
for value, count in rating_counts.items():
    percentage = (count / len(df)) * 100
    print(f"Value: {value}, Count: {count}, Percentage: {percentage:.2f}%")


total_letters = 0
total_words = 0
all_words = []

for text in df['review_description']:
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    all_words.extend(words)
    total_words += len(words)

    # Calculate total letters
    for word in words:
        total_letters += len(word)

# Calculate averages
average_letters = total_letters / len(df)
average_words = total_words / len(df)

# Calculate the vocabulary size
unique_words = set(all_words)
vocab_size = len(unique_words)

print(f"Average letters per review: {average_letters:.2f}")
print(f"Average words per review: {average_words:.2f}")
print(f"Vocabulary size: {vocab_size}")
