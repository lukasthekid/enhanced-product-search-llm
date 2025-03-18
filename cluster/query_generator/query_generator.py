import itertools
import json
import time

import numpy as np
import pandas as pd
import psutil
import torch
import transformers
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    TrainerCallback


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
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
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
        
    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)


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


if __name__ == "__main__":
    print('run main')
    data = pd.read_csv('query_product.tsv', sep='\t')
    # define train and test set
    unique_product_ids = data['product_id'].unique()
    # Split product_ids into train and test sets (90% train, from which we use another 10% for evaluation, 5% test for hyperparamter tuning)
    train_product_ids, test_product_ids = train_test_split(unique_product_ids, test_size=0.05, random_state=119)
    train_df = data[data['product_id'].isin(train_product_ids)]
    print(f'Train the model with {len(train_product_ids)} products and a total data shape of: {train_df.shape}')
    generator = BartQueryGenerator(train_df)
    generator.train(num_train_epochs=10, output_dir='models/results/bart')
    generator.save_model('models/fine-tuned-bart')