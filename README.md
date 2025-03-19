
# Enhanced Product Search Powered by LLMs

This repository contains two sets of code: one developed locally and another made using the TU WIEN GPU Cluster.

> **Note:** Some required datasets are not included in the repository due to their large size. You can download them from [this link](https://trec-product-search.github.io/2024.html).

When referring to file names in our code, please consult the following mapping table:

- **collection.tsv**: [Collection (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Corpus-v0.1/resolve/main/data/trec/collection.trec.gz)
- **2023_test_queries.tsv**: [2023 Test Queries (TREC Format)](https://huggingface.co/datasets/trec-product-search/product-search-2023-queries)
- **qid2query.tsv**: [Query2QueryID](https://huggingface.co/datasets/trec-product-search/Product-Search-Corpus-v0.1/resolve/main/data/qid2query.tsv)
- **train.jsonl**: [Train Triples JSONL](https://huggingface.co/datasets/trec-product-search/Product-Search-Triples/resolve/main/train.jsonl.gz)
- **2023test.qrel**: [2023 Test QREL (TREC Format)](https://huggingface.co/datasets/trec-product-search/product-search-2023-queries)
- **dev.qrels**: [Dev QRELS (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/resolve/main/data/dev/product-search-dev.qrels.gz)
- **train.qrel**: [Train QRELS (TREC Format)](https://huggingface.co/datasets/trec-product-search/Product-Search-Qrels-v0.1/resolve/main/data/train/product-search-train.qrels.gz)

If the data is not available, we upload everything we used to [HuggingFace](https://huggingface.co/datasets/lukastuwien/enhanced-product-search-llm/tree/main):


---

## Local Development

Inside the `local` folder, you'll find a subfolder called `notebooks`. This is where you can execute data preprocessing and explore the collection. It is essential to download the datasets and place them into the `/data` folder. The preprocessed data — generated by running `preprocessing.ipynb` — is subsequently used in later scripts. Preprocessed datasets will be referred to as `p_{originalFileName}` in future scripts. 
Locally we use Python 3.11 and a requirements.txt is also included in the folder

Configuration paths example:

```python
COLLECTION_PATH = '../data/collection.tsv'  
COLLECTION_OUTPUT_PATH = '../data/p_collection.tsv'  
COLLECTION_OUTPUT_PATH_SMALL = '../data/p_collection_small.tsv'  
QREL_TRAIN_PATH = '../data/QREL/train.qrels'  
QREL_DEV_PATH = '../data/QREL/dev.qrels'  
PASSAGES_PATH = '../data/train.jsonl'  
PASSAGES_OUTPUT_PATH = '../data/p_train.jsonl'  
QUERY_PRODUCT_PATH = '../data/query_product.tsv'  
QUERY_PATH = '../data/qid2query.tsv'
```

### BM25 Baseline

The BM25 baseline code is located under `application/service/model`. Specifically:
- In `main.py`, the baseline model is built.
- In `eval.py`, the baseline model is evaluated.

---

## Cluster Development

The majority of the work was carried out on the SLURM Cluster via cron jobs or Jupyter notebooks. To run the cluster-specific code, navigate to the `/cluster` folder.
On the cluster we use Python 3.9.21, a matching requirements.txt is also in the folder

**Dataset Setup:**  
You'll need to copy or move certain datasets into this directory as required. We've marked the necessary locations with empty files bearing the same names as the datasets. Use the `PreComputeEmbeddings` notebook to load the preprocessed collection, compute the embeddings, and save them to a pickle file—this file will be essential for subsequent steps. Note that the QREL folder also requires datasets.

### Build Query Generator (BART)

The `job.sh` script is used to submit various Python files as jobs on the cluster using SBATCH. Follow these steps:

1. **Train the BART Model:**  
   First, use the `job.sh` script to run `query_generator.py`, which trains a new BART model on the collection dataset to generate synthetic queries. After training, a folder containing the trained model will be created.  
   *Note:* The trained model is approximately 6GB in size and is not included in the repository. You will need to train the model on your own to obtain a fine-tuned BART model.

If you do not want to train the model you can dowload out model here (just unzip the folder in /query_generator directory): https://huggingface.co/datasets/lukastuwien/enhanced-product-search-llm/blob/main/gg_models.zip

2. **Hyperparameter Tuning:**  
   Once you have a model, use the `job.sh` script to perform hyperparameter tuning. This process will take some time, but it will ultimately print out the best configurations for your model. We have included sample log files to illustrate what the output should look like.

3. **Generate Synthetic Datasets:**  
   After tuning the model, use the `main.py` file to generate synthetic datasets comprising products and their corresponding synthetic search queries. Pre-generated synthetic sets are available in the `synth_set` folder for your reference.  
   *Note:* If a `.parquet` file is requested, it refers to a synthetic dataset containing embeddings for both the query and product description. You can use the same script that computed the collection embeddings here. Due to their large size, these embedded datasets are not included in the repository.

4. **Exploration and Visualization:**  
   Finally, run the Jupyter notebooks in this folder to explore the synthetic datasets and to visualize the model's performance during training.

### Build Recommender System

Next, navigate to the `/two_tower` folder to build the two-tower recommender systems fine-tuned on the synthetic queries. Key points:

- **Main Script:**  
  The primary script is `two-tower.py`. As mentioned in the thesis, we applied a cosine similarity filter and compared various models.  
  You can adjust the filter within the Python file.

- **Training:**  
  Use the `job_two_tower.sh` script to train your version of the two-tower model.
  
If you dont want to train the model you can find our models we used during the thesis here (just unzip the folder in /two_tower directory): https://huggingface.co/datasets/lukastuwien/enhanced-product-search-llm/blob/main/tt_models.zip

- **Architecture Exploration:**  
  Refer to the `ModelConfig` notebook, which explores different model architectures and helped us settle on the architecture specified in the Python file.  
  The `history` folder contains logs of the performance scores from various models during training, and the `Prot_Model` notebook provides comparative visualizations of these models.

### Evaluate Neural Retrievers

We have also included the all-MiniLM model as a baseline. To evaluate neural retrievers, follow these steps:

1. **Embedding Generation:**  
   Run the designated notebook to generate embeddings using the pretrained model.

2. **Model Evaluation:**  
   Once you have the embeddings, execute the `Evaluation.ipynb` notebook.  
   *Note:* You will need either a fine-tuned Neural Retriever from the previous stage or one of our pre-trained models (available for download) to complete the evaluation.

![image](/cluster/two_tower/styled_evaluation_metrics_chart%20(1).png)

## License

This master thesis, including all code, documentation, and data, is licensed under
the [Apache License 2.0](LICENSE).

This license was chosen to:
- Ensure academic recognition through attribution requirements
- Provide patent protection for novel methods
- Allow maximum reusability while requiring attribution
- Support both academic and potential commercial applications
