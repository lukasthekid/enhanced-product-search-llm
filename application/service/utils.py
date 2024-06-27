# text_preprocessor.py

import jsonlines
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

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

class RougeScore:
    def __init__(self, avg_rouge1, avg_rouge2, avg_rougeL, avg_rougeLsum):
        self.avg_rouge1 = avg_rouge1
        self.avg_rouge2 = avg_rouge2
        self.avg_rougeL = avg_rougeL
        self.avg_rougeLsum = avg_rougeLsum

    def to_dict(self):
        return {
            'avg_rouge1': self.avg_rouge1,
            'avg_rouge2': self.avg_rouge2,
            'avg_rougeL': self.avg_rougeL,
            'avg_rougeLsum': self.avg_rougeLsum
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            avg_rouge1=data.get('avg_rouge1', 0.0),
            avg_rouge2=data.get('avg_rouge2', 0.0),
            avg_rougeL=data.get('avg_rougeL', 0.0),
            avg_rougeLsum=data.get('avg_rougeLsum', 0.0)
        )

    def __repr__(self):
        return (f"RougeScore(avg_rouge1={self.avg_rouge1}, avg_rouge2={self.avg_rouge2}, "
                f"avg_rougeL={self.avg_rougeL}, avg_rougeLsum={self.avg_rougeLsum})")


def average_rouge_scores(rouge_scores: [RougeScore]) -> RougeScore:
    # Extract values into a numpy array
    rouge_array = np.array(
        [[score.avg_rouge1, score.avg_rouge2, score.avg_rougeL, score.avg_rougeLsum] for score in rouge_scores])

    # Calculate the averages using numpy
    average_rouge_scores = np.mean(rouge_array, axis=0)

    # Create a dictionary with the overall average scores
    overall_averages = RougeScore(average_rouge_scores[0], average_rouge_scores[1], average_rouge_scores[2],
                                  average_rouge_scores[3])

    return overall_averages


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


class SemanticHelper:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=False)


