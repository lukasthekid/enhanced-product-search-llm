import concurrent.futures
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from application.service.model.query_generator import BartQueryGenerator


class DatasetCreator:
    def __init__(self, model_path: str, product_descriptions: [str], batch_size=16, num_queries=5,
                 max_workers=8):
        self.device = self._get_device()
        self.model: BartQueryGenerator = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.product_descriptions = product_descriptions
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

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(0, len(self.product_descriptions), self.batch_size):
                batch_descriptions = self.product_descriptions[i:i + self.batch_size]
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

    def _create_positive_negative_samples(self) -> pd.DataFrame:
        random.seed(11925939)
        # Create positive samples
        samples = []
        for product, queries in self.product_to_queries.items():
            for query in queries:
                samples.append({
                    'query': query,
                    'product_description': product,
                    'relevance_label': 1
                })
        # Create negative samples by pairing queries with unrelated product descriptions
        all_descriptions = list(self.product_to_queries.keys())
        for product, queries in self.product_to_queries.items():
            unrelated_descriptions = [desc for desc in all_descriptions if desc != product]
            for query in queries:
                # Randomly select an unrelated product description
                negative_description = random.choice(unrelated_descriptions)
                samples.append({
                    'query': query,
                    'product_description': negative_description,
                    'relevance_label': 0
                })

        return pd.DataFrame(samples)

