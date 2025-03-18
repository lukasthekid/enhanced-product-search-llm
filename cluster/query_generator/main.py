import time
import numpy as np
import pandas as pd
import itertools
import json
import torch
import transformers
import concurrent.futures
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration
import random

class BartQueryGenerator:
    def __init__(self, data: pd.DataFrame = None, max_input_length=1024, max_output_length=64, device: str = None):
        self.device = torch.device(device) if device is not None else self._get_device()
        print(f'Model is on {self.device.type}')
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.data = data
        self.train_dataset = None
        self.val_dataset = None

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


    def generate_query(self, description, num_queries=5, num_beams=5, no_repeat_ngram=2, top_k=50,
                       top_p=0.95, temperature=0.8, length=1.2) -> [str]:
        self.model.eval()

        inputs = self.tokenizer(description, return_tensors='pt', max_length=self.max_input_length, truncation=True,
                                padding='max_length').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.max_output_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=True,
                no_repeat_ngram_size=no_repeat_ngram,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                length_penalty=length,
                num_return_sequences=num_queries
            )

        queries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        queries = list(dict.fromkeys(queries))
        return [q for q in queries if q]

    def generate_queries_batch(self, descriptions, num_queries=5, num_beams=5, no_repeat_ngram=2, top_k=50,
                               top_p=0.95, temperature=0.8, length=1.2, batch_size=16) -> dict:
        self.model.eval()

        all_queries = {}

        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]

            batch_queries = []
            for description in batch_descriptions:
                queries = self.generate_query(
                    description, num_queries=num_queries, num_beams=num_beams, no_repeat_ngram=no_repeat_ngram,
                    top_k=top_k, top_p=top_p, temperature=temperature, length=length
                )
                batch_queries.append(queries)

            for desc, queries in zip(batch_descriptions, batch_queries):
                all_queries[desc] = queries

        return all_queries

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, load_directory):
        self.model = BartForConditionalGeneration.from_pretrained(load_directory).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(load_directory)




class DatasetCreator:
    def __init__(self, model_path: str, product_texts: [str], batch_size=16, num_queries=5,
                 max_workers=4, device: str = None):
        self.model: BartQueryGenerator = BartQueryGenerator(device=device)
        self.model.load_model(model_path)
        self.product_texts = product_texts
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.max_workers = max_workers
        self.product_to_queries = self._product_to_query_parallel_batch()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _product_to_query_parallel_batch(self) -> dict:
        product_to_queries = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(self.product_texts), self.batch_size):
                batch_descriptions = self.product_texts[i:i + self.batch_size]
                futures.append(
                    executor.submit(self.model.generate_queries_batch, batch_descriptions, num_queries=self.num_queries,
                                    batch_size=self.batch_size))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                               desc="Generate Queries in Parallel Batches"):
                try:
                    result = future.result()
                    product_to_queries.update(result)
                except Exception as e:
                    print(f"Exception occurred: {e}")

        return product_to_queries

    def create_data_frame(self, negative_pairs=True) -> pd.DataFrame:
        random.seed(11925939)
        # Create positive and negative samples
        samples = []
        all_descriptions = list(self.product_to_queries.keys())
        for product, queries in tqdm(self.product_to_queries.items(), desc="Building Positive and Negative Passages"):
            unrelated_descriptions = [desc for desc in all_descriptions if desc != product]
            for query in queries:
                if negative_pairs:
                    # Randomly select an unrelated product description
                    negative_description = random.choice(unrelated_descriptions)
                    samples.append({
                        'query': query,
                        'product_text': negative_description,
                        'relevance_label': 0
                    })
                samples.append({
                    'query': query,
                    'product_text': product,
                    'relevance_label': 1
                })

        return pd.DataFrame(samples)
    



def build_synthetic_dataset(model_path='models/fine-tuned-bart', save_path='synthetic_positive_pairs.tsv'):
    collection = pd.read_csv('../p_collection.tsv', sep='\t')
    collection = collection.sample(frac=0.4, random_state=119)
    product_texts = collection['product_text'].values.tolist()
    negative_pairs = False
    #split in 4 equally sized sub arrays to speed up the process with multiple jobs
    #subarray_size = len(product_texts) // 4
    #subarrays = [product_texts[i:i + subarray_size] for i in range(0, len(product_texts), subarray_size)]
    #product_texts = subarrays[3]
    print(f'Generating queries for {len(product_texts)} products')
    start = time.time()
    creator = DatasetCreator(model_path, product_texts=product_texts, max_workers=4)
    df = creator.create_data_frame(negative_pairs=negative_pairs)
    print(f'Generating {df.shape[0]} Queries took {time.time() - start} seconds')

    df = pd.merge(df, collection[['id', 'product_text']])
    df.to_csv(save_path, sep='\t', index=False)


#REMOVE COMMENTS IF NEEDED
#print('run main')
#build_synthetic_dataset()