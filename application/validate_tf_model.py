import tensorflow as tf
from sentence_transformers import SentenceTransformer
from tensorflow.keras.layers import Input, Dense, Dot, Lambda, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
import time
import pandas as pd
import numpy as np
import faiss
import math

QUERY_TEST = '../data/train/2023_test_queries.tsv'
QREL_TEST = '../data/train/QREL/2023test.qrel'
COLLECTION = '../data/train/p_collection.tsv'
INDEX_GTE_FAISS = '../models/product_gte-large-en.index'
INDEX_TT_FAISS = '../models/product_two-tower.index'

BM25_path = '../models/bm25_baseline.pkl'
MODEL_WEIGHTS = '../models/fine-tuned-two-tower.pth'
ROBERTA_PATH = 'roberta-base'
GTE_ENG_MODEL = 'Alibaba-NLP/gte-large-en-v1.5'
KERAS_WEIGHTS = '../models/best_model_weights.keras'

def build_two_tower_model() -> tf.keras.models.Model:
    query_input = Input(shape=(1024,), name='query_input')
    product_input = Input(shape=(1024,), name='product_input')

    # Step 2: Build the Tower Model (Dense Layers for Query and Product Encoders)
    def build_tower(input_layer, prefix=''):
        x = Dense(512, activation='relu', name=prefix + '_first_embedding')(input_layer)
        x = Dense(256, activation='relu', name=prefix + '_second_embedding')(x)
        x = Dense(128, activation='relu', name=prefix + '_third_embedding')(x)
        x = Dense(64, activation='relu', name=prefix + '_last_embedding')(x)
        x = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1), name=prefix + '_normalizing',
                   output_shape=(64,))(x)
        return x

    # Apply the tower model to both inputs
    query_tower = build_tower(query_input, prefix='query')
    product_tower = build_tower(product_input, prefix='product')

    # Lambda layer to compute similarity
    similarity = Dot(axes=1, normalize=True, name='cosine_similarity')([query_tower, product_tower])

    return Model(inputs=[query_input, product_input], outputs=similarity)

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



def produce_ground_truth(qid: int, qrel: pd.DataFrame, collection: pd.DataFrame) -> [int]:
    df = qrel[(qrel['qid'] == qid) & (qrel['docid'].isin(collection['id'].values))]
    df = df.sort_values(by='relevance_score', ascending=False)
    return df.set_index('docid')['relevance_score'].to_dict()


# Assuming qrel is a DataFrame with columns ['qid', 'docid', 'relevance_score']
# recs is a DataFrame with the recommended documents, assuming it has at least a column 'id'
def produce_y_pred(qid: int, topk: pd.DataFrame, qrel: pd.DataFrame, ascending=False) -> dict:
    df = qrel[(qrel['qid'] == qid) & (qrel['docid'].isin(topk['id'].values))]
    df = df.copy()  # Create a copy to avoid the SettingWithCopyWarning
    df.loc[:, 'rank_score'] = topk.set_index('id').loc[df['docid'], 'score'].values
    df.sort_values(by='rank_score', ascending=ascending, inplace=True)
    return df.set_index('docid')['relevance_score'].to_dict()


def evaluate_two_tower(query_df: pd.DataFrame, qrel_df: pd.DataFrame, model, index,
                       collection: pd.DataFrame, sentence_transformer: SentenceTransformer, k=10, device='cpu') -> (
        float, float, float):
    ndcg = []
    precision = []
    recall = []

    for _, row in query_df.iterrows():
        if row.text is np.nan:
            continue
        y_true = produce_ground_truth(int(row.qid), qrel_df, collection)
        # Predict relevance scores
        input_data_query = [
            np.array(sentence_transformer.encode([str(row.text)])),
            np.array([np.array([0] * 1024)]),
        ]
        query_embedding = model.predict(input_data_query)
        # Perform the search
        D, I = index.search(query_embedding, collection.shape[0])  # D is distances, I is indices
        # Get the rows from documents corresponding to the indices
        top_documents = collection.iloc[I[0]].reset_index(drop=True)
        # Assign the score column using .loc to avoid the SettingWithCopyWarning
        top_documents.loc[:, 'score'] = D[0]
        y_pred = produce_y_pred(int(row.qid), top_documents, qrel_df, ascending=True)
        ndcg.append(normalized_discounted_cumulative_gain(y_pred))
        precision.append(precision_at_k(y_pred, y_true, k))
        recall.append(recall_at_k(y_pred, y_true, k))

    return np.mean(ndcg), np.mean(precision), np.mean(recall)


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

    index = faiss.read_index(INDEX_TT_FAISS)
    sentence_transformer = SentenceTransformer(GTE_ENG_MODEL, trust_remote_code=True)
    model = build_two_tower_model()
    model.load_weights(KERAS_WEIGHTS)
    query_model = Model(inputs=[model.get_layer('query_input').output, model.get_layer('product_input').output],
                        outputs=model.get_layer('query_normalizing').output
                        )
    score = evaluate_two_tower(qid2query_df, qrel_df, query_model, index, collection, sentence_transformer, k=k)
    print(f'NDCG score: {score[0]}')
    print(f"Precision@{k}: {score[1]}")
    print(f"Recall@{k}: {score[2]}")
