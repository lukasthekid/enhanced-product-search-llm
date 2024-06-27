import json
import os
import time

import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split

from application.service.model.query_generator import BartQueryGenerator
from service.evaluation import evaluate_bm25
from service.model.bm25 import BM25Okapi
from service.utils import TextPreprocessor, SemanticHelper

config = json.load(open("../config.json"))
os.environ['HF_TOKEN_READ'] = config['HF_TOKEN_READ']

print('Loading The product collection...')
start = time.time()
collection = pd.read_csv('../data/train/p_collection_small.tsv', sep='\t')
print(f'took {time.time() - start} seconds \n')
collection['title'] = collection['title'].fillna("")
preprocessor = TextPreprocessor()

qrel_df = pd.read_csv('../data/train/QREL/dev.qrels', sep='\t', names=['qid', '0', 'docid', 'relevance_score'],
                      header=None)
query_df = pd.read_csv('../data/train/qid2query.tsv', sep='\t', names=['qid', 'text'], header=None)
common_qids = set(qrel_df['qid']).intersection(set(query_df['qid']))
qrel_df = qrel_df[qrel_df['qid'].isin(common_qids)]
query_df = query_df[query_df['qid'].isin(common_qids)]


def build_baseline():
    query = 'black running shoe for men nike'
    collection['text'] = collection['title'] + " " + collection['description']
    corpus = collection['text'].tolist()
    # Preprocess the corpus
    print('Preprocess the Product Collection')
    start = time.time()

    # tokenized_corpus = [preprocessor.preprocess(doc) for doc in corpus]

    # with open("../data/train/tokenized_corpus.json", "w") as outfile:
    #    json.dump(tokenized_corpus, outfile)

    with open("../data/train/tokenized_corpus.json", 'r') as openfile:
        tokenized_corpus = json.load(openfile)

    print(f'Preprocessing finished in {time.time() - start} seconds \n')

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
        bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=params['k1'], b=params['b'], epsilon=params['epsilon'])
        score = evaluate_bm25(query_df.sample(n=10, random_state=119), qrel_df, bm25, collection)
        if score > best_score:
            best_score = score
            best_params = params

    print("Best Score:", best_score)
    print("Best Parameters:", best_params)
    print(f'Took {time.time() - start} seconds \n')

    # build and save best BM25 model
    start = time.time()
    bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=best_params['k1'], b=best_params['b'],
                     epsilon=best_params['epsilon'])
    print(f'Best BM25 initialized in {time.time() - start} seconds')
    tokenized_query = preprocessor.preprocess(query)
    start = time.time()
    print(f'Evaluate BM25 Model...')
    print(evaluate_bm25(query_df.sample(n=10, random_state=119), qrel_df, bm25, collection))
    print(f'Took {time.time() - start} seconds')

    bm25.save_model('../bm25_baseline.pkl')
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
        generator.train(num_train_epochs=2)
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


if __name__ == "__main__":
    print('run main')
    build_query_generator()
