import time

import pandas as pd
from transformers import AutoModel, AutoTokenizer

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from sentence_transformers import SentenceTransformer

from application.service.model.two_tower import load_faiss_index, build_two_tower_model
from service.evaluation import evaluate_bm25, evaluate_hf_encoder, evaluate_two_tower
from service.model.bm25 import BM25Okapi

QUERY_TEST = '../data/train/2023_test_queries.tsv'
QREL_TEST = '../data/train/QREL/2023test.qrel'
COLLECTION = '../data/train/p_collection.tsv'
INDEX_GTE_FAISS = '../models/product_gte-large-en.index'
INDEX_TT_FAISS = '../models/product_two-tower.index'

BM25_path = '../models/bm25_baseline.pkl'
MODEL_WEIGHTS = '../models/fine-tuned-two-tower.pth'
ROBERTA_PATH = 'roberta-base'
GTE_ENG_MODEL = 'Alibaba-NLP/gte-large-en-v1.5'
KERAS_MODEL = '../models/best_model_weights.keras'



def evaluate_bm25_main():
    bm25 = BM25Okapi.load_model(BM25_path)

    score = evaluate_bm25(qid2query_df, qrel_df, bm25, collection, k=k)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@{k}: {score[1]}")
    print(f"Recall@{k}: {score[2]}")

def evaluate_gte_large_en():
    index = load_faiss_index(INDEX_GTE_FAISS)
    tokenizer = AutoTokenizer.from_pretrained(GTE_ENG_MODEL)
    model = AutoModel.from_pretrained(GTE_ENG_MODEL, trust_remote_code=True)
    score = evaluate_hf_encoder(qid2query_df, qrel_df, model, tokenizer, index, collection, k=k)
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

    index = load_faiss_index(INDEX_TT_FAISS)
    sentence_transformer = SentenceTransformer(GTE_ENG_MODEL, trust_remote_code=True)
    model = build_two_tower_model()
    model.load_weights(KERAS_MODEL)
    query_model = Model(inputs=[model.get_layer('query_input').output, model.get_layer('product_input').output],
                        outputs=model.get_layer('query_normalizing').output
                        )
    score = evaluate_two_tower(qid2query_df, qrel_df, query_model, index, collection, sentence_transformer, k=k)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@{k}: {score[1]}")
    print(f"Recall@{k}: {score[2]}")

