import evaluate
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

from .model.bm25 import BM25
from .model.two_tower import BiEncoderModel, get_top_n_index
from .utils import TextPreprocessor, TopKHelper, SemanticHelper, RougeScore

preprocessor = TextPreprocessor()
helper = TopKHelper()


# Function to evaluate BM25 model
def evaluate_bm25(query_df: pd.DataFrame, qrel_df: pd.DataFrame, bm25: BM25, collection: pd.DataFrame, k=10) -> (
        float, float, float):
    ndcg = []
    precision = []
    recall = []

    for _, row in query_df.iterrows():
        if row.text is np.nan:
            continue
        y_true = helper.produce_ground_truth(int(row.qid), qrel_df, collection)
        # Predict relevance scores
        query_tokens = preprocessor.preprocess(row.text)
        topk_df = bm25.get_top_n(query_tokens, collection, n=0)
        y_pred = helper.produce_y_pred(int(row.qid), topk_df, qrel_df)
        ndcg.append(normalized_discounted_cumulative_gain(y_pred))
        precision.append(precision_at_k(y_pred, y_true, k))
        recall.append(recall_at_k(y_pred, y_true, k))

    return np.mean(ndcg), np.mean(precision), np.mean(recall)

def evaluate_two_tower(query_df: pd.DataFrame, qrel_df: pd.DataFrame, model: BiEncoderModel, index, collection: pd.DataFrame, k=10, device='cpu') -> (
        float, float, float):
    ndcg = []
    precision = []
    recall = []

    for _, row in query_df.iterrows():
        if row.text is np.nan:
            continue
        y_true = helper.produce_ground_truth(int(row.qid), qrel_df, collection)
        # Predict relevance scores
        query_embedding = model.encode_query(str(row.text), device=device).cpu().detach().numpy()
        topk_df = get_top_n_index(index, query_embedding, collection, k=collection.shape[0])
        y_pred = helper.produce_y_pred(int(row.qid), topk_df, qrel_df, ascending=True)
        ndcg.append(normalized_discounted_cumulative_gain(y_pred))
        precision.append(precision_at_k(y_pred, y_true, k))
        recall.append(recall_at_k(y_pred, y_true, k))

    return np.mean(ndcg), np.mean(precision), np.mean(recall)


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


def normalized_discounted_cumulative_gain(temp_set, p=5):
    dc_gain = 0
    idc_gain = 0
    for idx, value in enumerate(temp_set.values()):
        pos = idx + 1
        dc_gain += value / math.log2(pos + 1)
        if pos == p:
            break
    for idx, value in enumerate(sorted(temp_set.values(), reverse=True)):
        pos = idx + 1
        idc_gain += value / math.log2(pos + 1)
        if pos == p:
            break
    return round(dc_gain / idc_gain, 5)


def precision_at_k(predicted_dict, ideal_dict, k):
    # Get the top K docids from the predicted results
    top_k_pred = list(predicted_dict.keys())[:k]
    # Count the number of relevant documents in the top K predicted results
    relevant_in_pred = sum([1 for docid in top_k_pred if ideal_dict.get(docid, 0) > 0])
    # Precision is the number of relevant documents divided by K
    return relevant_in_pred / k


def recall_at_k(predicted_dict, ideal_dict, k):
    # Get the top K docids from the predicted results
    top_k_pred = list(predicted_dict.keys())[:k]
    # Count the total number of relevant documents in the ideal results
    total_relevant = sum([1 for score in ideal_dict.values() if score > 0])
    # Count the number of relevant documents in the top K predicted results
    relevant_in_pred = sum([1 for docid in top_k_pred if ideal_dict.get(docid, 0) > 0])
    # Recall is the number of relevant documents in top K divided by the total number of relevant documents
    return relevant_in_pred / total_relevant if total_relevant > 0 else 0
