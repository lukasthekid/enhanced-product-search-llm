import faiss
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dot, Lambda, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model


def build_two_tower_model() -> tf.keras.models.Model:
    query_input = Input(shape=(1024,), name='query_input')
    product_input = Input(shape=(1024,), name='product_input')

    # Step 2: Build the Tower Model (Dense Layers for Query and Product Encoders)
    def build_tower(input_layer, prefix=''):
        x = Dense(512, activation='relu', name=prefix + '_first_embedding')(input_layer)
        x = Dense(256, activation='relu', name=prefix + '_second_embedding')(x)
        x = Dense(128, activation='relu', name=prefix + '_third_embedding')(x)
        x = Dense(64, activation='relu', name=prefix + '_last_embedding')(x)
        x = Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1), name=prefix + '_normalizing',
                   output_shape=(64,))(x)
        return x

    # Apply the tower model to both inputs
    query_tower = build_tower(query_input, prefix='query')
    product_tower = build_tower(product_input, prefix='product')

    # Lambda layer to compute similarity
    similarity = Dot(axes=1, normalize=True, name='cosine_similarity')([query_tower, product_tower])

    return Model(inputs=[query_input, product_input], outputs=similarity)


def load_faiss_index(index_file):
    return faiss.read_index(index_file)


def get_top_n_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, documents: pd.DataFrame, k=10):
    # Perform the search
    D, I = index.search(query_embedding, k)  # D is distances, I is indices
    # Get the rows from documents corresponding to the indices
    top_documents = documents.iloc[I[0]].reset_index(drop=True)
    # Assign the score column using .loc to avoid the SettingWithCopyWarning
    top_documents.loc[:, 'score'] = D[0]
    return top_documents
