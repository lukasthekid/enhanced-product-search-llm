import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # Ensure this is imported
from transformers import RobertaTokenizer, RobertaModel


class BiEncoderModel(torch.nn.Module):
    def __init__(self, encoder_model):
        super(BiEncoderModel, self).__init__()
        self.query_encoder = RobertaModel.from_pretrained(encoder_model)
        self.product_encoder = RobertaModel.from_pretrained(encoder_model)

    def forward(self, query_input, product_input):
        query_output = self.query_encoder(**query_input).last_hidden_state
        product_output = self.product_encoder(**product_input).last_hidden_state

        # Mean pooling over all token embeddings
        query_embedding = query_output[:, 0, :]
        product_embedding = product_output[:, 0, :]

        return F.normalize(query_embedding, p=2, dim=1), F.normalize(product_embedding, p=2, dim=1)

    def encode_query(self, query: str, device):
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        query_encoding = tokenizer(query, return_tensors='pt', max_length=256, padding='max_length',
                                   truncation=True)

        # Move input to the same device as the model
        query_encoding = {key: val.to(device) for key, val in query_encoding.items()}
        query_output = self.query_encoder(**query_encoding).last_hidden_state
        query_embedding = query_output[:, 0, :]
        return F.normalize(query_embedding, p=2, dim=1)


def load_model(model_weights_path, model_name, device):
    model = BiEncoderModel(model_name)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def load_faiss_index(index_file):
    return faiss.read_index(index_file)


def get_top_n_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, documents: pd.DataFrame, k=10):
    # Perform the search
    D, I = index.search(query_embedding, k)  # D is distances, I is indices
    # Get the rows from documents corresponding to the indices
    top_documents = documents.iloc[I[0]].copy()  # Make an explicit copy of the DataFrame slice
    # Assign the score column using .loc to avoid the SettingWithCopyWarning
    top_documents.loc[:, 'score'] = D[0]
    return top_documents
