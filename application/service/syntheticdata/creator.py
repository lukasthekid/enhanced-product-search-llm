import concurrent.futures
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BartTokenizer

from application.service.model.query_generator import BartQueryGenerator


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

    def create_data_frame(self) -> pd.DataFrame:
        random.seed(11925939)
        # Create positive and negative samples
        samples = []
        all_descriptions = list(self.product_to_queries.keys())
        for product, queries in tqdm(self.product_to_queries.items(), desc="Building Positive and Negative Passages"):
            unrelated_descriptions = [desc for desc in all_descriptions if desc != product]
            for query in queries:
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
