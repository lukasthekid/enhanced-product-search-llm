from main import BartQueryGenerator
import evaluate
import math
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sentence_transformers import SentenceTransformer
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import os

MODEL_PATH = 'models/fine-tuned-bart'
POSTIVE_PAIRS = 'query_product.tsv'

model = BartQueryGenerator(device='cuda')
model.load_model(MODEL_PATH)

data = pd.read_csv(POSTIVE_PAIRS, sep='\t')
# define train and test set
unique_product_ids = data['product_id'].unique()
# Split product_ids into train and test sets (90% train, from which we use another 10% for evaluation, 5% test for hyperparamter tuning)
train_product_ids, test_product_ids = train_test_split(unique_product_ids, test_size=0.05, random_state=119)
# half of test set for hyper paramter tuning
tuning_ids = test_product_ids[:len(test_product_ids) // 2]
print(f'Tuning the model in {len(tuning_ids)} products')
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
helper = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def calculate_semantic_similarity(description, queries) -> [float, np.array]:
    # Get the embedding for the product description
    description_embedding = helper.encode(description, convert_to_tensor=False)
    # Get the embeddings for the generated queries
    query_embeddings = [helper.encode(query, convert_to_tensor=False) for query in queries]
    similarities = cosine_similarity([description_embedding], query_embeddings)[0]
    # Calculate the average similarity
    average_similarity = np.mean(similarities)
    return average_similarity, similarities


def calculate_rouge(reference: str, candidates: [str]):
    rouge = evaluate.load("rouge")
    rouge_scores = [rouge.compute(predictions=[c], references=[reference]) for c in candidates]
    rouge_array = np.array([
        [score['rouge1'], score['rouge2'], score['rougeL'], score['rougeLsum']]
        for score in rouge_scores
    ])
    avg_rouge_scores = np.mean(rouge_array, axis=0)

    # Create a dictionary with the average scores
    average_rouge_scores = {'rouge1':avg_rouge_scores[0], 'rouge2': avg_rouge_scores[1], 'rougeL': avg_rouge_scores[2], 'rougeLsum':avg_rouge_scores[3]}

    return average_rouge_scores


def compute_average(scores_list):
    if not scores_list:
        return {}
    
    sum_scores = defaultdict(float)
    count = len(scores_list)
    
    for score_dict in scores_list:
        for key, value in score_dict.items():
            sum_scores[key] += value
    
    average_scores = {key: sum_scores[key] / count for key in sum_scores}
    return average_scores


# parallel setup

def evaluate_params(num_beams, top_p, temperature, test_product_ids, data):
    cos_sim_scores = []
    rouge_scores = []
    for pid in test_product_ids:
        selected_product = data[data['product_id'] == pid]
        ground_truth = selected_product['query'].values.tolist()
        if len(ground_truth) != 1:
            continue
        text = selected_product['product_text'].values[0]
        synthetic_queries = model.generate_query(
            text,
            num_queries=5,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature
        )
        cos_sim_scores.append(calculate_semantic_similarity(text, synthetic_queries)[0])
        rouge_scores.append(calculate_rouge(ground_truth, synthetic_queries))
    # Compute aggregated scores
    overall_rouge_score = compute_average(rouge_scores)
    avg_rouge = sum([score/len(overall_rouge_score) for score in overall_rouge_score.values()])
    
    overall_cos_sim_score = np.mean(cos_sim_scores)
    score = 1.5 * avg_rouge + overall_cos_sim_score
    return (num_beams, top_p, temperature, score)

def hyperparameter_tuning(test_product_ids, data):
    num_beams_range = [5, 7, 10]
    top_p_range = [0.8, 0.9, 0.95]
    temperature_range = [0.7, 0.8, 0.9, 1.0]
    param_grid = list(itertools.product(num_beams_range, top_p_range, temperature_range))
    
    results = Parallel(n_jobs=3)(
        delayed(evaluate_params)(
            nb, tp, temp, test_product_ids, data
        ) for nb, tp, temp in param_grid
    )
    
    # Find best params
    best_params = None
    best_score = float('-inf')
    for (nb, tp, temp, score) in results:
        if score > best_score:
            best_score = score
            best_params = (nb, tp, temp)
    
    return best_params



start = time.time()
best_score = hyperparameter_tuning(tuning_ids, data)

print(f'Best Scores: num_beams:{best_score[0]}, top_p:{best_score[1]}, temperature:{best_score[2]}')
print(f'took {time.time() - start} seconds')