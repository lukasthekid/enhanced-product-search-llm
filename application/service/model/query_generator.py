from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

class QueryGeneratorBART:
    def __init__(self):
        # Load pre-trained BART model and tokenizer
        self.dataset = None
        self.model_name = 'facebook/bart-large'
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)

    def generate_synthetic_queries(self, product_description):
        # Prepare input text for the model
        input_text = f"Generate search queries for the product: {product_description}"

        # Tokenize input text
        inputs = self.tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)

        # Generate synthetic queries
        outputs = self.model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)

        # Decode generated queries
        synthetic_queries = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return synthetic_queries

    def prepare_fine_tuning_data(self, data):
        inputs = []
        outputs = []

        for item in data:
            query = item['query']
            for passage in item.get('positive_passages', []):
                product_description = passage['text']
                inputs.append(f"Generate search queries for the product: {product_description}")
                outputs.append(query)

        dataset = Dataset.from_dict({'input_text': inputs, 'target_text': outputs})
        self.dataset = dataset
        return dataset

    def tokenize_function(self, examples):
        model_inputs = self.tokenizer(examples['input_text'], max_length=512, truncation=True)
        labels = self.tokenizer(examples['target_text'], max_length=50, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def fine_tuning(self):
        tokenized_dataset = self.dataset.map(self.tokenize_function, batched=True)

        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
        )

        # Initialize Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )

        # Train model
        trainer.train()









