import time

import pandas as pd
from service.evaluation import evaluate_bm25
from service.model.bm25 import BM25Okapi

QUERY_TEST = '../data/train/2023_test_queries.tsv'
QREL_TEST = '../data/train/QREL/2023test.qrel'
COLLECTION = '../data/train/p_collection.tsv'

BM25_path = '../models/bm25_baseline.pkl'



def evaluate_bm25_main():
    bm25 = BM25Okapi.load_model(BM25_path)

    score = evaluate_bm25(qid2query_df, qrel_df, bm25, collection, k=k)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@{k}: {score[1]}")
    print(f"Recall@{k}: {score[2]}")


if __name__ == "__main__":
    print('run evaluation script')
    k = 15

    start = time.time()
    collection = pd.read_csv(COLLECTION, sep='\t')
    qid2query_df = pd.read_csv(QUERY_TEST, sep='\t', names=['qid', 'text'], header=None)
    qrel_df = pd.read_csv(QREL_TEST, sep='\t', names=['qid', '0', 'docid', 'relevance_score'], header=None)
    common_qids = set(qrel_df['qid']).intersection(set(qid2query_df['qid']))
    qrel_df = qrel_df[qrel_df['qid'].isin(common_qids)]
    qid2query_df = qid2query_df[qid2query_df['qid'].isin(common_qids)]
    print(f'loading all data took {time.time() - start} seconds')
    print(f'Testing the model with {qid2query_df.shape} queries')

    evaluate_bm25_main()

