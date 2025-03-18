import itertools
import json
import time

import numpy as np
import pandas as pd
import psutil
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    TrainerCallback

from local.application.service.evaluation import calculate_semantic_similarity, calculate_rouge, average_cosine_distance
from local.application.service.utils import RougeScore, average_rouge_scores, SemanticHelper


class BartQueryGenerator:
    def __init__(self, data: pd.DataFrame = None, max_input_length=1024, max_output_length=64, device: str = None):
        self.model_name = 'facebook/bart-large'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.device = torch.device(device) if device is not None else self._get_device()
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

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

    def _load_data(self):
        descriptions = self.data['product_text'].values.tolist()
        queries = self.data['query'].values.tolist()
        dataset = Dataset.from_dict({'product_description': descriptions, 'search_query': queries})
        return dataset

    def _tokenize_data(self, dataset):
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['product_description'],
                max_length=self.max_input_length,
                truncation=True,
                padding='max_length'
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['search_query'],
                    max_length=self.max_output_length,
                    truncation=True,
                    padding='max_length'
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def _split_data(self, tokenized_datasets):
        train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
        self.train_dataset = train_test_split['train']
        self.val_dataset = train_test_split['test']

    def train(self, num_train_epochs=3, learning_rate=2e-5, output_dir='./results/bart'):
        dataset = self._load_data()
        tokenized_datasets = self._tokenize_data(dataset)
        self._split_data(tokenized_datasets)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=learning_rate,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=True,
            predict_with_generate=True,  # Use generation during evaluation
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[ResourceTrackerCallback(log_interval=10)]
        )

        print('Start training')
        start = time.time()
        trainer.train()
        print(f'Finished training after {time.time() - start} seconds')

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

    def hyperparameter_tuning(self, test_product_ids: [str], data: pd.DataFrame, semanticHelper: SemanticHelper):
        best_params = None
        best_score = float('-inf')
        # Define the ranges for hyperparameters
        num_beams_range = [5, 7]
        top_p_range = [0.9, 0.95]
        temperature_range = [0.8, 0.9]
        # Create all combinations of hyperparameters
        param_grid = list(itertools.product(num_beams_range, top_p_range, temperature_range))
        for params in tqdm(param_grid, desc="Hyperparameter Tuning"):
            # print('run with', params)
            num_beams, top_p, temperature = params
            diversity_scores = []
            cos_sim_scores = []
            rouge_scores = []
            for id in test_product_ids:
                selected_product = data[data['product_id'] == id]
                ground_truth = selected_product['query'].values.tolist()
                if len(ground_truth) != 1:
                    continue
                text = selected_product['product_text'].values[0]
                synthetic_queries = self.generate_query(text, num_queries=5, num_beams=num_beams,
                                                        top_p=top_p,
                                                        temperature=temperature)
                cos_sim_scores.append(calculate_semantic_similarity(text, synthetic_queries, semanticHelper)[0])
                rouge_scores.append(calculate_rouge(ground_truth, synthetic_queries))
                diversity_scores.append(average_cosine_distance(synthetic_queries, semanticHelper))
            # each score ranges from 0 to 1
            overall_rouge_score: RougeScore = average_rouge_scores(rouge_scores)
            overall_cos_sim_score = np.mean(cos_sim_scores)
            overall_diversity_score = np.mean(diversity_scores)
            avg_rouge = (overall_rouge_score.avg_rouge1 + overall_rouge_score.avg_rouge2 +
                         overall_rouge_score.avg_rougeL + overall_rouge_score.avg_rougeLsum) / 4
            score = 1.5 * avg_rouge + 1.0 * overall_cos_sim_score
            # print(score)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params

    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, load_directory):
        self.model = BartForConditionalGeneration.from_pretrained(load_directory).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(load_directory)



class ResourceTrackerCallback(TrainerCallback):
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.training_log = []
        self.start_time = None
        self.process = psutil.Process()

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self._log_resources(state, initial=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            self._log_resources(state)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        self._log_resources(state, metrics=metrics)

    def on_train_end(self, args, state, control, **kwargs):
        self._log_resources(state)
        self._save_log(args.output_dir)

    def _log_resources(self, state, initial=False, metrics=None):
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        memory_usage = self.process.memory_info().rss / (1024 ** 2)  # in MB
        cpu_usage = self.process.cpu_percent(interval=None)

        log_entry = {
            'step': state.global_step,
            'epoch': state.epoch,
            'elapsed_time': elapsed_time,
            'memory_usage_mb': memory_usage,
            'cpu_usage_percent': cpu_usage,
        }
        if metrics:
            log_entry.update(metrics)
        if initial:
            log_entry['initial'] = True

        self.training_log.append(log_entry)
        print(log_entry)

    def _save_log(self, output_dir):
        log_file = f'{output_dir}/training_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.training_log, f, indent=4)
