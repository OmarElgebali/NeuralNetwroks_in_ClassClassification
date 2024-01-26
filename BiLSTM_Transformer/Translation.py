import demoji
from googletrans import Translator
from tqdm import tqdm
import pandas as pd
import time

tqdm.pandas()
translator = Translator()


def translate_english_to_arabic(input_string):
    listOfWords = input_string.split()
    arabic_range = (0x0600, 0x06FF)
    english_range = (0x0020, 0x007E)

    for index, word in enumerate(listOfWords):
        if arabic_range[0] <= ord(word[0]) <= arabic_range[1]:
            continue
        else:
            while True:
                try:
                    translatedWord = translator.translate(word, dest='ar').text
                    break
                except Exception as e:
                    if "timed out" in str(e).lower():
                        time.sleep(1)
            listOfWords[index] = translatedWord
    returnedString = ' '.join(listOfWords)
    return returnedString


def translate_en_to_ar_with_emoji(text):
    emojis = demoji.findall(text)
    translated_text = text

    for emoji, desc in emojis.items():
        while True:
            try:
                translated_emoji = translator.translate(desc, src='en', dest='ar').text
                translated_text = translated_text.replace(emoji, f' {translated_emoji}')
                break
            except Exception as e:
                if "timed out" in str(e).lower():
                    time.sleep(1)

    while True:
        try:
            translated_text = translator.translate(translated_text, src='en', dest='ar').text
            break
        except Exception as e:
            if "timed out" in str(e).lower():
                time.sleep(1)

    return translated_text


def translate_en_to_ar(text):
    translated_text = text

    while True:
        try:
            translated_text = translator.translate(translated_text, src='en', dest='ar').text
            break
        except Exception as e:
            if "timed out" in str(e).lower():
                time.sleep(1)

    return translated_text


df = pd.read_csv('train_3ashawa2ya_100_rows.csv')
# df = df.sample(n=200, random_state=42)
# df.to_csv('train_3ashawa2ya_100_rows.csv', encoding='utf-8-sig')
df['Arabic_Translation'] = df['review_description'].progress_apply(lambda row: translate_english_to_arabic(row))
df.to_csv('train_translated_100_rows_Random2.csv', encoding='utf-8-sig')
