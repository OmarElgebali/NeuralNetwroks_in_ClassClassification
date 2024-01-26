
import time
import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.offline import init_notebook_mode, iplot
from tashaphyne.stemming import ArabicLightStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, f1_score, accuracy_score, recall_score, roc_auc_score, \
    make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score
from farasa.stemmer import FarasaStemmer
import re
import emoji
# import langid
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from langdetect import detect
from nltk.corpus import stopwords
from googletrans import Translator
# init_notebook_mode(connected=True)
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()
df = pd.read_csv('Train_Dataset.csv')
# print("df BEFORE duplicate", df)
# df.review_description.duplicated().sum()
# df.drop(df[df.review_description.duplicated() == True].index, axis=0, inplace=True)
# # print("df after duplicate", df)
#
# df = df.rename({'rating(1 postive 0 neutral -1 negative': 'label'}, axis=1)
# # df['rating'] = df['rating'].replace({-1: 'negative', 0: 'neutral', 1: 'positive'})
# # print("After target renaming", df)
#
# df.review_description = df.review_description.astype(str)
# df.review_description = df.review_description.progress_apply(
#     lambda x: re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
# df.review_description = df.review_description.progress_apply(lambda x: x.replace('؛', "", ))

# print("df After punctuation removal", df)

stopWords = list(set(stopwords.words("arabic")))  ## To remove duplictes and return to list again
# Some words needed to work with to will remove
for word in ['لا', 'لكن', 'ولكن']:
    stopWords.remove(word)

# df.review_description = df.review_description.progress_apply(
#     lambda x: " ".join([word for word in x.split() if word not in stopWords]))

emojis = {
    "🙂": "يبتسم",
    "🥰": "حب",
    "💔": "قلب حزين",
    "❤️": "حب",
    "❤": "حب",
    "😍": "حب",
    "😭": "يبكي",
    "😢": "حزن",
    "😔": "حزن",
    "♥": "حب",
    "💜": "حب",
    "😅": "يضحك",
    "🙁": "حزين",
    "💕": "حب",
    "💙": "حب",
    "😞": "حزين",
    "😊": "سعادة",
    "👏": "يصفق",
    "👌": "احسنت",
    "😴": "ينام",
    "😀": "يضحك",
    "😌": "حزين",
    "🌹": "وردة",
    "🙈": "حب",
    "😄": "يضحك",
    "😐": "محايد",
    "✌": "منتصر",
    "✨": "نجمه",
    "🤔": "تفكير",
    "😏": "يستهزء",
    "😒": "يستهزء",
    "🙄": "ملل",
    "😕": "عصبية",
    "😃": "يضحك",
    "🌸": "وردة",
    "😓": "حزن",
    "💞": "حب",
    "💗": "حب",
    "😑": "منزعج",
    "💭": "تفكير",
    "😎": "ثقة",
    "💛": "حب",
    "😩": "حزين",
    "💪": "عضلات",
    "👍": "موافق",
    "🙏🏻": "رجاء طلب",
    "😳": "مصدوم",
    "👏🏼": "تصفيق",
    "🎶": "موسيقي",
    "🌚": "صمت",
    "💚": "حب",
    "🙏": "رجاء طلب",
    "💘": "حب",
    "🍃": "سلام",
    "☺": "يضحك",
    "🐸": "ضفدع",
    "😶": "مصدوم",
    "✌️": "مرح",
    "✋🏻": "توقف",
    "😉": "غمزة",
    "🌷": "حب",
    "🙃": "مبتسم",
    "😫": "حزين",
    "😨": "مصدوم",
    "🎼 ": "موسيقي",
    "🍁": "مرح",
    "🍂": "مرح",
    "💟": "حب",
    "😪": "حزن",
    "😆": "يضحك",
    "😣": "استياء",
    "☺️": "حب",
    "😱": "كارثة",
    "😁": "يضحك",
    "😖": "استياء",
    "🏃🏼": "يجري",
    "😡": "غضب",
    "🚶": "يسير",
    "🤕": "مرض",
    "‼️": "تعجب",
    "🕊": "طائر",
    "👌🏻": "احسنت",
    "❣": "حب",
    "🙊": "مصدوم",
    "💃": "سعادة مرح",
    "💃🏼": "سعادة مرح",
    "😜": "مرح",
    "👊": "ضربة",
    "😟": "استياء",
    "💖": "حب",
    "😥": "حزن",
    "🎻": "موسيقي",
    "✒": "يكتب",
    "🚶🏻": "يسير",
    "💎": "الماظ",
    "😷": "وباء مرض",
    "☝": "واحد",
    "🚬": "تدخين",
    "💐": "ورد",
    "🌞": "شمس",
    "👆": "الاول",
    "⚠️": "تحذير",
    "🤗": "احتواء",
    "✖️": "غلط",
    "📍": "مكان",
    "👸": "ملكه",
    "👑": "تاج",
    "✔️": "صح",
    "💌": "قلب",
    "😲": "مندهش",
    "💦": "ماء",
    "🚫": "خطا",
    "👏🏻": "برافو",
    "🏊": "يسبح",
    "👍🏻": "تمام",
    "⭕️": "دائره كبيره",
    "🎷": "ساكسفون",
    "👋": "تلويح باليد",
    "✌🏼": "علامه النصر",
    "🌝": "مبتسم",
    "➿": "عقده مزدوجه",
    "💪🏼": "قوي",
    "📩": "تواصل معي",
    "☕️": "قهوه",
    "😧": "قلق و صدمة",
    "🗨": "رسالة",
    "❗️": "تعجب",
    "🙆🏻": "اشاره موافقه",
    "👯": "اخوات",
    "©": "رمز",
    "👵🏽": "سيده عجوزه",
    "🐣": "كتكوت",
    "🙌": "تشجيع",
    "🙇": "شخص ينحني",
    "👐🏽": "ايدي مفتوحه",
    "👌🏽": "بالظبط",
    "⁉️": "استنكار",
    "⚽️": "كوره",
    "🕶": "حب",
    "🎈": "بالون",
    "🎀": "ورده",
    "💵": "فلوس",
    "😋": "جائع",
    "😛": "يغيظ",
    "😠": "غاضب",
    "✍🏻": "يكتب",
    "🌾": "ارز",
    "👣": "اثر قدمين",
    "❌": "رفض",
    "🍟": "طعام",
    "👬": "صداقة",
    "🐰": "ارنب",
    "☂": "مطر",
    "⚜": "مملكة فرنسا",
    "🐑": "خروف",
    "🗣": "صوت مرتفع",
    "👌🏼": "احسنت",
    "☘": "مرح",
    "😮": "صدمة",
    "😦": "قلق",
    "⭕": "الحق",
    "✏️": "قلم",
    "ℹ": "معلومات",
    "🙍🏻": "رفض",
    "⚪️": "نضارة نقاء",
    "🐤": "حزن",
    "💫": "مرح",
    "💝": "حب",
    "🍔": "طعام",
    "❤︎": "حب",
    "✈️": "سفر",
    "🏃🏻‍♀️": "يسير",
    "🍳": "ذكر",
    "🎤": "مايك غناء",
    "🎾": "كره",
    "🐔": "دجاجة",
    "🙋": "سؤال",
    "📮": "بحر",
    "💉": "دواء",
    "🙏🏼": "رجاء طلب",
    "💂🏿 ": "حارس",
    "🎬": "سينما",
    "♦️": "مرح",
    "💡": "قكرة",
    "‼": "تعجب",
    "👼": "طفل",
    "🔑": "مفتاح",
    "♥️": "حب",
    "🕋": "كعبة",
    "🐓": "دجاجة",
    "💩": "معترض",
    "👽": "فضائي",
    "☔️": "مطر",
    "🍷": "عصير",
    "🌟": "نجمة",
    "☁️": "سحب",
    "👃": "معترض",
    "🌺": "مرح",
    "🔪": "سكينة",
    "♨": "سخونية",
    "👊🏼": "ضرب",
    "✏": "قلم",
    "🚶🏾‍♀️": "يسير",
    "👊": "ضربة",
    "◾️": "وقف",
    "😚": "حب",
    "🔸": "مرح",
    "👎🏻": "لا يعجبني",
    "👊🏽": "ضربة",
    "😙": "حب",
    "🎥": "تصوير",
    "👉": "جذب انتباه",
    "👏🏽": "يصفق",
    "💪🏻": "عضلات",
    "🏴": "اسود",
    "🔥": "حريق",
    "😬": "عدم الراحة",
    "👊🏿": "يضرب",
    "🌿": "ورقه شجره",
    "✋🏼": "كف ايد",
    "👐": "ايدي مفتوحه",
    "☠️": "وجه مرعب",
    "🎉": "يهنئ",
    "🔕": "صامت",
    "😿": "وجه حزين",
    "☹️": "وجه يائس",
    "😘": "حب",
    "😰": "خوف و حزن",
    "🌼": "ورده",
    "💋": "بوسه",
    "👇": "لاسفل",
    "❣️": "حب",
    "🎧": "سماعات",
    "📝": "يكتب",
    "😇": "دايخ",
    "😈": "رعب",
    "🏃": "يجري",
    "✌🏻": "علامه النصر",
    "🔫": "يضرب",
    "❗️": "تعجب",
    "👎": "غير موافق",
    "🔐": "قفل",
    "👈": "لليمين",
    "™": "رمز",
    "🚶🏽": "يتمشي",
    "😯": "متفاجأ",
    "✊": "يد مغلقه",
    "😻": "اعجاب",
    "🙉": "قرد",
    "👧": "طفله صغيره",
    "🔴": "دائره حمراء",
    "💪🏽": "قوه",
    "💤": "ينام",
    "👀": "ينظر",
    "✍🏻": "يكتب",
    "❄️": "تلج",
    "💀": "رعب",
    "😤": "وجه عابس",
    "🖋": "قلم",
    "🎩": "كاب",
    "☕️": "قهوه",
    "😹": "ضحك",
    "💓": "حب",
    "☄️ ": "نار",
    "👻": "رعب",
    "❎": "خطء",
    "🤮": "حزن",
    '🏻': "احمر"
}

emoticons_to_emoji = {
    ":)": "🙂",
    ":(": "🙁",
    "xD": "😆",
    ":=(": "😭",
    ":'(": "😢",
    ":'‑(": "😢",
    "XD": "😂",
    ":D": "🙂",
    "♬": "موسيقي",
    "♡": "❤",
    "☻": "🙂",
}


def checkemojie(text):
    emojistext = []
    for char in text:
        if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
            emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
    return " ".join(emojistext)


def emojiTextTransform(text):
    cleantext = re.sub(r'[^\w\s]', '', text)
    return cleantext + " " + checkemojie(text)


def detect_language(input_string):
    # Define Unicode ranges for Arabic and English characters
    arabic_range = (0x0600, 0x06FF)
    english_range = (0x0020, 0x007E)

    # Check if the string contains Arabic characters
    arabic_chars = any(ord(char) in range(arabic_range[0], arabic_range[1] + 1) for char in input_string)

    # Check if the string contains English characters
    english_chars = any(ord(char) in range(english_range[0], english_range[1] + 1) for char in input_string)

    if arabic_chars and not english_chars:
        return "Arabic"
    elif english_chars and not arabic_chars:
        return "English"
    else:
        return "Mixed"


def translate_english_to_arabic(input_string):
    # Define Unicode ranges for Arabic and English characters
    listOfWords = input_string.split()
    arabic_range = (0x0600, 0x06FF)
    english_range = (0x0020, 0x007E)

    translator = Translator()

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


df.review_description = df.review_description.progress_apply(lambda x: emojiTextTransform(x))

# df.review_description = df.review_description.progress_apply(
#     lambda x: ''.join([word for word in x if not word.isdigit()]))

df.review_description = df.review_description.progress_apply(lambda x: translate_english_to_arabic(x))

df.review_description = df.review_description.progress_apply(
    lambda x: " ".join([ArabicLightStemmer().light_stem(word) for word in x.split()]))

df.to_csv('train_dataset_preprocessed.csv', encoding='utf-8-sig')
