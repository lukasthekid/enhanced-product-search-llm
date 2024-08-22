import json
import multiprocessing
import os
import time

import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split

from application.service.model.query_generator import BartQueryGenerator
from application.service.syntheticdata.creator import DatasetCreator
from service.evaluation import evaluate_bm25
from service.model.bm25 import BM25Okapi
from service.utils import TextPreprocessor, SemanticHelper

config = json.load(open("../config.json"))
os.environ['HF_TOKEN_READ'] = config['HF_TOKEN_READ']
collection_path = '../data/train/p_collection.tsv'
qrel_dev_path = '../data/train/QREL/dev.qrels'
qid_to_query_path = '../data/train/qid2query.tsv'
baseline_tokenization_path = '../data/train/tokenized_corpus.json'

print('Loading The product collection...')
start = time.time()
# collection = pd.read_csv(collection_path, sep='\t')
print(f'took {time.time() - start} seconds \n')
preprocessor = TextPreprocessor()


def build_baseline(preprocessing=True, tuning=True):
    collection = pd.read_csv(collection_path, sep='\t')
    qid2query_df = pd.read_csv(qid_to_query_path, sep='\t', names=['qid', 'text'], header=None)
    qrel_df = pd.read_csv(qrel_dev_path, sep='\t', names=['qid', '0', 'docid', 'relevance_score'], header=None)
    common_qids = set(qrel_df['qid']).intersection(set(qid2query_df['qid']))
    qrel_df = qrel_df[qrel_df['qid'].isin(common_qids)]
    qid2query_df = qid2query_df[qid2query_df['qid'].isin(common_qids)]

    if preprocessing:
        preprocessor = TextPreprocessor()
        corpus = collection['product_text'].tolist()
        print('Preprocess the Product Collection')
        start = time.time()
        tokenized_corpus = [preprocessor.preprocess(doc) for doc in corpus]
        with open(baseline_tokenization_path, "w") as outfile:
            json.dump(tokenized_corpus, outfile)
        print(f'Preprocessing finished in {time.time() - start} seconds \n')

    with open(baseline_tokenization_path, 'r') as openfile:
        tokenized_corpus = json.load(openfile)

    print(f'collection shape: {collection.shape}, tokenized_corpus shape {len(tokenized_corpus)} (size should be same)')
    if tuning:
        start = time.time()
        # Define the parameter grid
        param_grid = {
            'k1': [1.2, 1.5, 1.8],
            'b': [0.6, 0.75, 0.9],
            'epsilon': [0.1, 0.25, 0.5]
        }

        # Grid search
        print(f'Start the Hyperparamter Tuning')
        best_score = 0
        best_params = None
        for params in ParameterGrid(param_grid):
            bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=params['k1'], b=params['b'],
                             epsilon=params['epsilon'])
            # tune based on ndcg score
            score = evaluate_bm25(qid2query_df.sample(n=10, random_state=119), qrel_df, bm25, collection)[0]
            if score > best_score:
                best_score = score
                best_params = params

        print("Best Score:", best_score)
        print("Best Parameters:", best_params)
        print(f'Took {time.time() - start} seconds \n')

        # build and save best BM25 model
        bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=best_params['k1'], b=best_params['b'],
                         epsilon=best_params['epsilon'])
    else:
        bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=1.2, b=0.9,
                         epsilon=0.1)

    print(f'Evaluate BM25 Model...')
    start = time.time()
    print(f'NDCG, Precision and Recall Scores: {evaluate_bm25(qid2query_df.sample(n=10, random_state=119), qrel_df, bm25, collection)}')
    print(f'Took {time.time() - start} seconds')

    bm25.save_model('../models/bm25_baseline.pkl')
    print('saved successfully')


def build_query_generator(train=False, evaluate=True):
    generator = BartQueryGenerator()
    data = pd.read_csv('../data/train/query_product.tsv', sep='\t')
    # define train and test set
    unique_product_ids = data['product_id'].unique()
    # Split product_ids into train and test sets (90% train, 10% test)
    train_product_ids, test_product_ids = train_test_split(unique_product_ids, test_size=0.05, random_state=119)
    train_df = data[data['product_id'].isin(train_product_ids)]

    if train:
        print(f'Train the model with {len(train_product_ids)} products and a total data shape of: {train_df.shape}')
        generator = BartQueryGenerator(train_df)
        generator.train(num_train_epochs=2, output_dir='../models/results/bart')
        generator.save_model('../models/fine-tuned-bart')
    else:
        generator.load_model('../models/fine-tuned-bart')
    if evaluate:
        tuning_ids = test_product_ids[:len(test_product_ids) // 2]
        print(f'Hyperparameter Tuning with {len(tuning_ids)} product descriptions the Model did not see before')
        start = time.time()
        best_score = generator.hyperparameter_tuning(tuning_ids, data, semanticHelper=SemanticHelper())
        print(f'Best Scores: num_beams:{best_score[0]}, top_p:{best_score[1]}, temperature:{best_score[2]}')
        print(f'took {time.time() - start} seconds')


def build_synthetic_dataset(model_path='../models/fine-tuned-bart', save_path='../data/synthetic_query_product.tsv'):
    collection = pd.read_csv('../data/train/p_collection_small.tsv', sep='\t')
    product_texts = collection['product_text'].values.tolist()
    print(f'Generating queries for {len(product_texts)} products')
    print(f'{multiprocessing.cpu_count()} Cores available')
    start = time.time()
    creator = DatasetCreator(model_path, product_texts=product_texts, max_workers=multiprocessing.cpu_count())
    df = creator.create_data_frame()
    print(f'Generating {df.shape[0] // 2} Queries took {time.time() - start} seconds')

    # df = pd.merge(df, collection[['id', 'product_text']])
    # df.to_csv(save_path, sep='\t', index=False)


if __name__ == "__main__":
    print('run main')
    build_baseline(preprocessing=False, tuning=False)
