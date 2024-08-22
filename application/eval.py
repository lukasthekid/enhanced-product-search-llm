import time

import pandas as pd
from transformers import AutoModel

from application.service.model.two_tower import load_model, load_faiss_index
from service.evaluation import evaluate_bm25, evaluate_two_tower
from service.model.bm25 import BM25Okapi

QUERY_TEST = '../data/train/2023_test_queries.tsv'
QREL_TEST = '../data/train/QREL/2023test.qrel'
COLLECTION = '../data/train/p_collection.tsv'
BM25_path = '../models/bm25_baseline.pkl'

MODEL_WEIGHTS = '../models/fine-tuned-two-tower.pth'
ROBERTA_PATH = 'roberta-base'
INDEX_FAISS = '../models/product_index.index'


def evaluate_bm25_main():
    bm25 = BM25Okapi.load_model(BM25_path)

    score = evaluate_bm25(qid2query_df, qrel_df, bm25, collection, k=10)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@10: {score[1]}")
    print(f"Recall@10: {score[2]}")


if __name__ == "__main__":
    print('run evaluation script')

    start = time.time()
    collection = pd.read_csv(COLLECTION, sep='\t')
    qid2query_df = pd.read_csv(QUERY_TEST, sep='\t', names=['qid', 'text'], header=None)
    qrel_df = pd.read_csv(QREL_TEST, sep='\t', names=['qid', '0', 'docid', 'relevance_score'], header=None)
    common_qids = set(qrel_df['qid']).intersection(set(qid2query_df['qid']))
    qrel_df = qrel_df[qrel_df['qid'].isin(common_qids)]
    qid2query_df = qid2query_df[qid2query_df['qid'].isin(common_qids)]
    print(f'loading all data took {time.time() - start} seconds')
    print(f'Testing the model with {qid2query_df.shape} queries')

    model = load_model(MODEL_WEIGHTS, ROBERTA_PATH, 'cpu')
    index = load_faiss_index(INDEX_FAISS)
    score = evaluate_two_tower(qid2query_df, qrel_df, model, index, collection, k=10)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@10: {score[1]}")
    print(f"Recall@10: {score[2]}")
