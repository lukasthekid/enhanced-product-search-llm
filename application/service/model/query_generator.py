import json
import time

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, \
    TrainingArguments
import torch
from datasets import Dataset
class BartQueryGenerator:
    def __init__(self, data, max_input_length=1024, max_output_length=512):
        # Load pre-trained BART model and tokenizer
        self.train_dataset = None
        self.val_dataset = None
        self.model_name = 'facebook/bart-large'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.data = data

    def _load_data(self):
        train_data = []
        for entry in self.data:
            for pos_passage in entry["positive_passages"]:
                train_data.append({
                    "query_id": entry["query_id"],
                    "query": entry["query"],
                    "product_description": pos_passage["text"],
                    "product_id": pos_passage["docid"]
                })
        descriptions = [item['product_description'] for item in train_data]
        queries = [item['query'] for item in train_data]
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
        #return tokenized_datasets.shuffle(seed=42).select(range(10))
        return tokenized_datasets

    def _split_data(self, tokenized_datasets):
        train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
        self.train_dataset = train_test_split['train']
        self.val_dataset = train_test_split['test']

    def train(self, num_train_epochs=3, learning_rate=2e-5):
        dataset = self._load_data()
        tokenized_datasets = self._tokenize_data(dataset)
        self._split_data(tokenized_datasets)

        training_args = Seq2SeqTrainingArguments(
            output_dir='./results',
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
        )
        print(f'start training')
        start = time.time()
        trainer.train()
        print(f'finished after {time.time() - start} seconds')

    def generate_query(self, description):
        self.model.eval()
        # Move model to CPU
        self.model.to('cpu')

        inputs = self.tokenizer(description, return_tensors='pt', max_length=self.max_input_length, truncation=True,
                                padding='max_length')
        # Ensure inputs are on the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(inputs['input_ids'], max_length=self.max_output_length, num_beams=5,
                                      early_stopping=True)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self, model_path, tokenizer_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)

    def load_model(self, model_path, tokenizer_path):
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = BartTokenizer.from_pretrained(tokenizer_path)


class Llama3QueryGenerator:

    def __init__(self, data, max_input_length=1024, max_output_length=512):
        # Load pre-trained BART model and tokenizer
        model_id = "meta-llama/Meta-Llama-3-8B"
        self.max_output_length = max_output_length
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )

    def generate_query(self, description):
        self.model.eval()
        # Move model to CPU
        self.model.to('cpu')
        content = "Generate me a suitable search query for the following product: " + description

        messages = [
            {"role": "system", "content": "You are a reverse search engine that gives search queries for product descriptions"},
            {"role": "user", "content": content},
        ]

        prompt = self.pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=self.max_output_length,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        return outputs








