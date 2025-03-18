#!/usr/bin/env python3
# train_retriever.py

import os
import gc
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.parquet as pq
from typing import Tuple, Generator, Dict, Text

# !nvidia-smi
# !nvcc --version

# Set random seed for reproducibility
SEED = 11925939
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Configure TensorFlow for optimal GPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # If using PyTorch elsewhere
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) Available: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Using CPU.")

print("TensorFlow version:", tf.__version__)

# Paths to data files
PARQUET_FILE_PATH = '../synth_set/synthetic_positive_pairs.parquet.gzip'
PCOLLECTION_FILE_PATH = '../p_collection.pkl'

# Hyperparameters
BATCH_SIZE = 2048
NUM_EPOCHS = 50
LEARNING_RATE = 0.1
DECAY_STEPS = 10000
DECAY_RATE = 0.96
PATIENCE = 10
MODEL_CHECKPOINT_PATH = 'best_model_tow_tower.keras'
MODEL_SAVING_PATH = 'model/weights'
HISTORY_JSON_PATH = 'training_history.json'  # Path to save the history JSON
SIMILARITY_THRESHOLD = 0.5  # Threshold to filter out low similarity pairs
TRAIN_SPLIT = 0.8
SEED = 119


def prepare_data(
        parquet_file_path: str,
        pcollection_file_path: str,
        batch_size: int,
        seed: int = 119,
        train_split: float = 0.8,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    pairs = pd.read_parquet(parquet_file_path)
    pairs = pd.merge(pairs, pd.read_pickle(pcollection_file_path), on='id')
    pairs.dropna(subset=['query', 'product_embedding'], inplace=True)
    pairs['product_embedding'] = pairs['product_embedding'].apply(np.array)
    pairs['query_embedding'] = pairs['query_embedding'].apply(np.array)
    pairs['cosine_similarity'] = pairs.apply(
        lambda row: cosine_similarity([row['product_embedding']], [row['query_embedding']])[0][0], axis=1)
    pairs = pairs[pairs['cosine_similarity'] >= 0.5]
    total_samples: int = len(pairs)
    unique_embeddings = pairs.groupby('id')['product_embedding'].first().values

    print('Compete Dataset Shape:', pairs.shape, 'candidates: {}', len(unique_embeddings))
    query_embeddings = pairs['query_embedding'].values
    product_embeddings = pairs['product_embedding'].values
    del pairs  # free memory
    dataset = tf.data.Dataset.from_tensor_slices({
        "query_embedding": np.stack(query_embeddings),
        "product_embedding": np.stack(product_embeddings)
    })
    candidates = tf.data.Dataset.from_tensor_slices(np.stack(unique_embeddings))

    shuffled_dataset = dataset.shuffle(total_samples, seed=seed, reshuffle_each_iteration=False)

    # Calculate split indices
    train_size = int(train_split * total_samples)
    print('training with', train_size, 'pairs')

    # Split the dataset
    train_dataset = shuffled_dataset.take(train_size)
    test_dataset = shuffled_dataset.skip(train_size)

    # Optimize dataset pipeline
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    del query_embeddings
    del product_embeddings
    del dataset
    del shuffled_dataset

    return train_dataset, test_dataset, candidates


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, input_dim=1024):
        super().__init__()
        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(tf.keras.layers.Input(shape=(input_dim,)))
        self.dense_layers.add(tf.keras.layers.Dense(128, activation='elu'))
        self.dense_layers.add(tf.keras.layers.Dense(128))
        self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1)))

    def call(self, inputs):
        return self.dense_layers(inputs)


class ProductModel(tf.keras.Model):
    """Model for encoding products."""

    def __init__(self, input_dim=1024):
        super().__init__()
        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(tf.keras.layers.Input(shape=(input_dim,)))
        self.dense_layers.add(tf.keras.layers.Dense(128, activation='elu'))
        self.dense_layers.add(tf.keras.layers.Dense(128))
        self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=-1)))

    def call(self, inputs):
        return self.dense_layers(inputs)


class TwoTowerModel(tfrs.models.Model):

    def __init__(self, candidates):
        super().__init__()
        self.query_model = QueryModel()
        self.product_model = ProductModel()
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=candidates.batch(128).map(self.product_model)
            ),
        )

    def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        query_embeddings = self.query_model(features["query_embedding"])
        product_embeddings = self.product_model(features["product_embedding"])

        return (
            query_embeddings,
            product_embeddings
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False):
        query_embeddings = self.query_model(features["query_embedding"])
        product_embeddings = self.product_model(features["product_embedding"])

        return self.task(
            query_embeddings, product_embeddings, compute_metrics=not training)


train_ds, test_ds, candidates = prepare_data(
    parquet_file_path=PARQUET_FILE_PATH,
    pcollection_file_path=PCOLLECTION_FILE_PATH,
    batch_size=BATCH_SIZE,
    seed=SEED,
    train_split=TRAIN_SPLIT
)

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    alpha=0.0
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model = TwoTowerModel(candidates)
model.compile(optimizer=optimizer)
# Use callbacks for early stopping and model checkpointing
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_factorized_top_k/top_100_categorical_accuracy', patience=PATIENCE)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_tow_tower.keras', monitor='val_factorized_top_k/top_100_categorical_accuracy', save_best_only=True)

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=NUM_EPOCHS,
    callbacks=[],
    verbose=1)

with open(HISTORY_JSON_PATH, 'w') as f:
    json.dump(history.history, f, indent=4)
print("Training history saved.")

model.save_weights(MODEL_SAVING_PATH, save_format='tf')
loaded_model = TwoTowerModel(candidates)
loaded_model.load_weights(MODEL_SAVING_PATH)

query = tf.random.uniform((1, 1024))
product = tf.random.uniform((1, 1024))

t_query, t_product = loaded_model({
    "query_embedding": np.array([query]),
    "product_embedding": np.array([product])
})

# Reshape tensors to 2D
tensor1_reshaped = tf.reshape(t_query, (1, 128)).numpy()
tensor2_reshaped = tf.reshape(t_product, (1, 128)).numpy()

# Compute cosine similarity
cosine_score = cosine_similarity(tensor1_reshaped, tensor2_reshaped)

print("Cosine Similarity Score (TEST):", cosine_score[0][0])