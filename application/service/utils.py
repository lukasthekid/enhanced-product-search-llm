# text_preprocessor.py

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import jsonlines

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



def read_jsonl_file(filepath):
    data = []
    with jsonlines.open(filepath) as reader:
        for obj in reader:
            data.append(obj)
    return data


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text) -> [str]:
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


class TopKHelper:

    def produce_ground_truth(self, query: {int, str}, qrel: pd.DataFrame) -> [int]:
        df = qrel[qrel['qid'] == query['id']]
        df = df.sort_values(by='relevance_score', ascending=False)
        return df['relevance_score'].values

    # Assuming qrel is a DataFrame with columns ['qid', 'docid', 'relevance_score']
    # recs is a DataFrame with the recommended documents, assuming it has at least a column 'id'
    def produce_y_pred(self, qid: int, recs: pd.DataFrame, qrel: pd.DataFrame) -> [int]:
        # Create a dictionary for quick lookup of relevance scores
        qrel_dict = qrel[qrel['qid'] == qid].set_index('docid')['relevance_score'].to_dict()

        # Generate the predictions list using the dictionary for quick lookups
        r = [qrel_dict.get(doc_id, 0) for doc_id in recs['id']]

        return r
