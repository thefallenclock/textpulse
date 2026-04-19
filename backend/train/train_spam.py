import pandas as pd
import os
import string
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Get absolute path to project root.
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# o/p: c:\Projects\textpulse.

# Joining the path till spam.csv
file_path = os.path.join(base_dir, "dataset", "spam.csv")
# o/p: c:\Projects\textpulse\dataset\spam.csv

# Load dataset.
df = pd.read_csv(file_path, encoding="latin-1")

# Show first 5 rows.
# print(df.head())
# o/p:      v1  ... Unnamed: 4
# o/p: 0   ham  ...        NaN
# o/p: 1   ham  ...        NaN
# o/p: 2  spam  ...        NaN
# o/p: 3   ham  ...        NaN
# o/p: 4   ham  ...        NaN
# o/p: 
# o/p: [5 rows x 5 columns]

# REMOVE EXTRA COLUMNS:
# before: Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='str')
df = df[['v1', 'v2']]
# print(df.columns)
# o/p: Index(['v1', 'v2'], dtype='str')

# RENAME THE COLUMNS:
df.columns = ['labels', 'text']
# print(df.columns)
# o/p: Index(['labels', 'text'], dtype='str')

# CONVERT LABELS:
df['labels'] = df['labels'].map({'ham': 0, 'spam':1})
# print(df.head())
# o/p:    labels                                               text
# o/p: 0       0  Go until jurong point, crazy.. Available only ...
# o/p: 1       0                      Ok lar... Joking wif u oni...
# o/p: 2       1  Free entry in 2 a wkly comp to win FA Cup fina...
# o/p: 3       0  U dun say so early hor... U c already then say...
# o/p: 4       0  Nah I don't think he goes to usf, he lives aro...

# REMOVE STOPWORDS:
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# TEXT CLEAN(into numbers):
def clean_text(text):
    text = text.lower() # convert to lowercase.
    text = text.translate(str.maketrans('', '', string.punctuation)) # removes punctuation.
    # after removing stopwords.
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# apply clean text.
df['text'] = df['text'].apply(clean_text)
# (before removing stopwords)
# print(df.head())
# o/p:    labels                                               text
# o/p: 0       0  go until jurong point crazy available only in ...
# o/p: 1       0                            ok lar joking wif u oni
# o/p: 2       1  free entry in 2 a wkly comp to win fa cup fina...
# o/p: 3       0        u dun say so early hor u c already then say
# o/p: 4       0  nah i dont think he goes to usf he lives aroun...

# (after removing stopwords)
# print(df.head())
# o/p:    labels                                               text
# o/p: 0       0  go jurong point crazy available bugis n great ...
# o/p: 1       0                            ok lar joking wif u oni
# o/p: 2       1  free entry 2 wkly comp win fa cup final tkts 2...
# o/p: 3       0                u dun say early hor u c already say
# o/p: 4       0        nah dont think goes usf lives around though

# CONVERT TEXT INTO NUMBERS(TF-IDF):
vectorizer = TfidfVectorizer()
# learns all unique words and converts text into numbers.
x = vectorizer.fit_transform(df['text']) # input(text as numbers).
y = df['labels'] # output(spam/ham).

