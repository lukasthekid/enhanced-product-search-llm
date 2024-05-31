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
Best Score: 0.9420475565916557
Best Parameters: {'b': 0.6, 'epsilon': 0.1, 'k1': 1.5}
Took 1303.495684146881 seconds 

top10 ndgc on 10 queries
Evaluate BM25 Model...
0.9420475565916557
Took 13.96976113319397 seconds
