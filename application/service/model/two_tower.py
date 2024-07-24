import os
import time

import nvidia_smi
import pandas as pd
import psutil
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_system_metrics():
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.used / (1024 ** 3)  # Convert to GB

        gpu_handles = [nvidia_smi.nvmlDeviceGetHandleByIndex(i) for i in range(nvidia_smi.nvmlDeviceGetCount())]
        gpu_info = []
        for handle in gpu_handles:
            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_info.append(dict({
                'memory_used': mem_info.used / (1024 ** 3),  # Convert to GB
                'memory_total': mem_info.total / (1024 ** 3),  # Convert to GB
                'utilization': utilization.gpu
            }))

        return dict({'cpu_usage': cpu_usage, 'memory_usage': memory_usage, 'gpu_info': gpu_info})
    except Exception:
        return dict({})


class TwoTowerModel(nn.Module):
    def __init__(self, model_name, device, q_max_length=64, p_max_length=512):
        super(TwoTowerModel, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.product_encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, queries, products):
        query_inputs = self.tokenizer(queries, return_tensors='pt', padding=True, truncation=True,
                                      max_length=self.q_max_length).to(self.device)
        product_inputs = self.tokenizer(products, return_tensors='pt', padding=True, truncation=True,
                                        max_length=self.p_max_length).to(self.device)

        query_embeddings = self.query_encoder(**query_inputs).last_hidden_state.mean(dim=1)
        product_embeddings = self.product_encoder(**product_inputs).last_hidden_state.mean(dim=1)

        similarities = (query_embeddings * product_embeddings).sum(dim=1)
        return similarities

    def compute_loss(self, similarities, labels):
        return self.loss(similarities, labels)


class RetrievalDataset(Dataset):
    def __init__(self, queries, products, labels):
        self.queries = queries
        self.products = products
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.products[idx], self.labels[idx]


class TwoTowerTrainer:
    def __init__(self, model_name: str, device='cuda', access_token=None):
        self.device = torch.device(device)
        self.model = TwoTowerModel(model_name, device=self.device)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _create_data_loader(self, train_data: pd.DataFrame, val_data: pd.DataFrame, batch_size: int) -> (
    DataLoader, DataLoader):
        # Extract relevant columns
        train_queries = train_data['query'].tolist()
        train_products = train_data['product_text'].tolist()
        train_labels = train_data['relevance_label'].tolist()

        val_queries = val_data['query'].tolist()
        val_products = val_data['product_text'].tolist()
        val_labels = val_data['relevance_label'].tolist()

        # Create DataLoader
        train_dataset = RetrievalDataset(train_queries, train_products, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = RetrievalDataset(val_queries, val_products, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def _train_epoch(self, model, dataloader, optimizer, scheduler, device):
        model.train()
        total_loss = 0
        for queries, products, labels in dataloader:
            labels = torch.tensor(labels, dtype=torch.float).to(device)
            optimizer.zero_grad()
            similarities = model(queries, products)
            loss = model.compute_loss(similarities, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for queries, products, labels in dataloader:
                labels = torch.tensor(labels, dtype=torch.float).to(device)
                similarities = model(queries, products)
                loss = model.compute_loss(similarities, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame, num_epochs=3, batch_size=16):
        train_dataloader, val_dataloader = self._create_data_loader(train_data, val_data, batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        history = []
        start_time = time.time()
        for epoch in tqdm(range(num_epochs), desc="Iterating through the EPOCHs"):
            train_loss = self._train_epoch(self.model, train_dataloader, optimizer, scheduler, self.device)
            val_loss = self.evaluate(self.model, val_dataloader, self.device)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            history.append({
                'epoch': epoch + 1,
                'training_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'batch_size': batch_size,
                'elapsed_time': time.time() - start_time,
                'system_metric': get_system_metrics()
            })
        return history

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)


class TwoTowerRetriever:
    def __init__(self, model_name: str, fined_tuned_weights_path: str, collection: pd.DataFrame = None,
                 max_length_query=64,
                 max_length_product=512,
                 device="cuda"):
        self.index = None
        self.max_length_query = max_length_query
        self.max_length_product = max_length_product
        self.device = torch.device(device)
        self.model_path = model_name
        self.model = TwoTowerModel.load_model(model_name)
        self.model.load_state_dict(torch.load(fined_tuned_weights_path))
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collection = collection

    def retrieve_top_k(self, query, k=10):
        if self.collection is None:
            raise Exception("Product Collection is not defined")

        self.model.eval()
        with torch.no_grad():
            query_inputs = self.tokenizer(query, return_tensors='pt', truncation=True, max_length=128).to(self.device)
            query_embedding = self.model.query_encoder(**query_inputs).last_hidden_state.mean(dim=1)

            product_texts = self.collection['product_text'].values.tolist()

            product_embeddings = []
            for product in product_texts:
                product_inputs = self.tokenizer(product, return_tensors='pt', truncation=True, max_length=128).to(self.device)
                product_embedding = self.model.product_encoder(**product_inputs).last_hidden_state.mean(dim=1)
                product_embeddings.append(product_embedding)

            product_embeddings = torch.cat(product_embeddings)
            similarities = (query_embedding * product_embeddings).sum(dim=1)

            top_k_indices = torch.argsort(similarities, descending=True)[:k]

        return [product_texts[i] for i in top_k_indices]
