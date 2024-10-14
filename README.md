# enhanced-product-search-llm

use the virtual environment on Python 3.11.9

using
``venv/bin/python --version``

``venv/bin/jupyter lab ``


changes in the BM25 implementation

Summary of Changes:
Initialization and Tokenization: Simplified the initialization logic and tokenization process.
IDF Calculation: Streamlined the _calc_idf method to make it more concise.
Score Calculation: Simplified the get_scores method, eliminating redundant calculations and making it clearer.
General Simplifications: Removed unnecessary comments and code to improve readability and maintain functionality.

times:
Loading The product collection...
took 15.826967239379883 seconds

Preprocess the Product Collection
Preprocessing finished in 1103.0511558055878 seconds 

Start the Hyperparamter Tuning
Best Score: 0.7686189999999999
Best Parameters: {'b': 0.9, 'epsilon': 0.1, 'k1': 1.2}
Took 2005.11181306839 seconds
Validation Set Score
NDCG, Precision and Recall Scores: (0.7686189999999999, 0.7799999999999999, 0.7795815295815296)

Test Baseline on Test Data
Testing the model with (186, 2) queries
NDCG score: 0.6637909139784945
Precision@5: 0.6881720430107527
Recall@5: 0.19493303684155178
Precision@10: 0.5806451612903226
Recall@10: 0.2577583666969151

Query Generator
small data model training 3 epochs took
finished after 62090.06482172012 seconds

Hyperparameter Tuning Query Generator
Best Scores: num_beams:5, top_p:0.95, temperature:0.8
took 242043.24642181396 seconds

Testing Pretrained GTE Encoder Model
Testing the model with (186, 2) queries
NDCG score: 0.635121935483871
Precision@5: 0.6849462365591398
Recall@5: 0.18533256727291184
Precision@10: 0.5978494623655914
Recall@10: 0.26196451221269507
Precision@15: 0.5462365591397849
Recall@15: 0.31576822070917676
