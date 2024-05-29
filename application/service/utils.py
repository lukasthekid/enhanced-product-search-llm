# text_preprocessor.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text):
        try:
            # Tokenize text
            tokens = word_tokenize(text)
        # Convert to lower case
            tokens = [word.lower() for word in tokens]
        # Remove punctuation
            tokens = [word for word in tokens if word.isalnum()]
        # Remove stop words
            tokens = [word for word in tokens if word not in self.stop_words]
        # Lemmatize tokens
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            return tokens
        except Exception:
            print(text, Exception)
