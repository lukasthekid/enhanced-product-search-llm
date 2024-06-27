import evaluate
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from .model.bm25 import BM25
from .utils import TextPreprocessor, TopKHelper, SemanticHelper, RougeScore

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


# Evaluate Query Generator
def calculate_bleu(reference: str, candidates: [str]) -> float:
    bleu = evaluate.load("bleu")
    bleu_scores = [bleu.compute(predictions=[c], references=[reference]) for c in candidates]
    return np.average(list(map(lambda x: x['bleu'], bleu_scores)))


def calculate_rouge(reference: str, candidates: [str]) -> [RougeScore]:
    rouge = evaluate.load("rouge")
    rouge_scores = [rouge.compute(predictions=[c], references=[reference]) for c in candidates]
    rouge_array = np.array([
        [score['rouge1'], score['rouge2'], score['rougeL'], score['rougeLsum']]
        for score in rouge_scores
    ])
    avg_rouge_scores = np.mean(rouge_array, axis=0)

    # Create a dictionary with the average scores

    average_rouge_scores = RougeScore(avg_rouge_scores[0], avg_rouge_scores[1], avg_rouge_scores[2],
                                      avg_rouge_scores[3])

    return average_rouge_scores


def calculate_semantic_similarity(description, queries, helper: SemanticHelper) -> [float, np.array]:
    # Get the embedding for the product description
    description_embedding = helper.get_embedding(description)
    # Get the embeddings for the generated queries
    query_embeddings = [helper.get_embedding(query) for query in queries]
    similarities = cosine_similarity([description_embedding], query_embeddings)[0]
    # Calculate the average similarity
    average_similarity = np.mean(similarities)
    return average_similarity, similarities


def average_cosine_distance(queries, helper: SemanticHelper):
    if len(queries) < 2:
        # Not enough queries to compute pairwise distances
        return 0.0
    embeddings = [helper.get_embedding(query) for query in queries]
    distances = cosine_distances(embeddings)
    upper_tri_distances = distances[np.triu_indices_from(distances, k=1)]

    if upper_tri_distances.size == 0:
        # No upper triangular distances to average
        return 0.0

    return upper_tri_distances.mean()