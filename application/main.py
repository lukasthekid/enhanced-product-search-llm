import json

import pandas as pd
from sklearn.model_selection import ParameterGrid

from application.service.model.query_generator import QueryGeneratorBART
from service.utils import TextPreprocessor, read_jsonl_file;
from service.model.bm25 import BM25Okapi
from service.evaluation import evaluate_bm25
import time


print('Loading The product collection...')
start = time.time()
collection = pd.read_csv('../data/train/p_collection_small.tsv', sep='\t')
print(f'took {time.time() - start} seconds \n')
collection['title'] = collection['title'].fillna("")
preprocessor = TextPreprocessor()

qrel_df = pd.read_csv('../data/train/QREL/dev.qrels', sep='\t', names=['qid', '0', 'docid', 'relevance_score'], header=None)
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
    bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=best_params['k1'], b=best_params['b'], epsilon=best_params['epsilon'])
    print(f'Best BM25 initialized in {time.time() - start} seconds')
    tokenized_query = preprocessor.preprocess(query)
    start = time.time()
    print(f'Evaluate BM25 Model...')
    print(evaluate_bm25(query_df.sample(n=10, random_state=119), qrel_df, bm25, collection))
    print(f'Took {time.time() - start} seconds')

    bm25.save_model('../bm25_baseline.pkl')
    print('saved successfully')


if __name__ == "__main__":

    print('run main')
    #build_baseline()
    # Extract product description from positive passage
    generator = QueryGeneratorBART()
    print('Inizialisation finished')
    start = time.time()
    data = [{'positive_passages':[{'text':"Product Description Read more Read more All-purpose Utility The Ram-Pro’s ten inch, ready to install Air Tire Wheels is the best replacement for your hand truck tires and wheels. It offers easy installation and optimum effort reducing performance. The only effort you need to make is to remove the old tires before sliding in the new ones and closing them with clips, nuts or cotter pins. That’s a perfectly installed and proper functioning hand truck without any hesitation or frustration. Superior Quality The high-quality heavy duty rubber will last very long. The air stem is on the outside so you can easily inflate the tires if needed. The double sealed bearings will evenly distribute the load on your vehicle and help to balance the loud noise evenly thereby making you push less and easier. The reduction in noise levels further helps to reduce stress levels. High End Specification The Ram-Pro Air Tires size is ten inches high and three inches wide with a hole diameter of 5/8 inches. The hub depth of the tires is 1-3/4 inches with double sealed bearings. With such high-end specifications, it can withstand a load capacity of 300 lbs. (approximately 136 Kgs). The pressure per square inch rating is 30 per tire. The tube type is a 2 ply 4.10/3.5-4. These air-filled tires are designed with raised grips to make your drive the smoothest possible one. Numerous Applications The air tires can be used for hand trucks, lawn mowers, yard wagons, air compressors, power washers, child's wagons, shopping carts, wood chippers, snow blowers, dollies, go karts, golf carts, tricycles, and much more. This assembly has proven to be ideal for air compressors, dollies, generators, pressure washers and various other auxiliary utility equipment. Read more Read more "}]}]
    #data = read_jsonl_file("../data/train/train.jsonl")
    print(f'Loading the json data took {time.time() - start} seconds')
    product_description = data[0]['positive_passages'][0]['text']
    print(product_description)

    # Generate synthetic queries
    synthetic_queries = generator.generate_synthetic_queries(product_description)
    print("Generated Query: ", synthetic_queries)

