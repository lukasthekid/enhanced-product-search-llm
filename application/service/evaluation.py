import math
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .model.bm25 import BM25
from .utils import TextPreprocessor, TopKHelper

preprocessor = TextPreprocessor()
helper = TopKHelper()



# Function to evaluate BM25 model
def evaluate_bm25(query_df: pd.DataFrame, qrel_df: pd.DataFrame, bm25: BM25, collection: pd.DataFrame, k=10):
    y_preds = []
    y_trues = []

    for _, row in query_df.iterrows():
        if row.text is np.nan:
            continue
        y_true = helper.produce_ground_truth({'id': row.qid, 'text': row.text}, qrel_df)[:k]

        # Predict relevance scores
        query_tokens = preprocessor.preprocess(row.text)

        y_pred = bm25.get_scores(query_tokens)
        collection['scores'] = y_pred
        recommendations = collection.sort_values(by='scores', ascending=False).head(k)
        y_pred = helper.produce_y_pred(int(row.qid), recommendations, qrel_df)

        # Pad y_true with zeros to ensure all lists have the same length
        padded_y_true = np.pad(y_true, (0, k - len(y_true)), 'constant', constant_values=0)

        y_preds.append(y_pred)
        y_trues.append(padded_y_true)

    # Convert lists to numpy arrays
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)

    # Calculate NDCG score
    ndcg = ndcg_score(y_trues, y_preds)

    return ndcg