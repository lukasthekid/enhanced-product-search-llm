
import pandas as pd
from service.utils import TextPreprocessor;
from service.model.bm25 import BM25Okapi
import time

print('Loading The product collection...')
start = time.time()
df = pd.read_csv('../data/train/p_collection_small.tsv', sep='\t')
print(f'took {time.time() - start} seconds \n')
df['title'] = df['title'].fillna("")
preprocessor = TextPreprocessor()

if __name__ == "__main__":
    query = 'black running shoe for men nike'
    df['text'] = df['title'] + " " + df['description']
    corpus = df['text'].tolist()
    # Preprocess the corpus
    print('Preprocess the Product Collection')
    start = time.time()
    tokenized_corpus = [preprocessor.preprocess(doc) for doc in corpus]
    print(f'Preprocessing finished in {time.time() - start} seconds \n')
    start = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f'BM25 initialized in {time.time() - start} seconds')
    tokenized_query = preprocessor.preprocess(query)
    start = time.time()
    scores = bm25.get_scores(tokenized_query)
    print(f'BM25 scores calculated in {time.time() - start} seconds')
    df['score'] = scores
    results = df.sort_values(by='score', ascending=False)
    print(results[['id', 'title', 'score']].head())
