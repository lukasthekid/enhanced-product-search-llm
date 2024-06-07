import json

import pandas as pd
from sklearn.model_selection import ParameterGrid

from application.service.model.query_generator import BartQueryGenerator, Llama3QueryGenerator
from service.utils import TextPreprocessor, read_jsonl_file
from service.model.bm25 import BM25Okapi
from service.evaluation import evaluate_bm25
import time


print('Loading The product collection...')
start = time.time()
collection = pd.read_csv('../data/train/p_collection_small.tsv', sep='\t')
print(f'took {time.time() - start} seconds \n')
collection['title'] = collection['title'].fillna("")
preprocessor = TextPreprocessor()

qrel_df = pd.read_csv('../data/train/QREL/dev.qrels', sep='\t', names=['qid', '0', 'docid', 'relevance_score'], header=None)
query_df = pd.read_csv('../data/train/qid2query.tsv', sep='\t', names=['qid', 'text'], header=None)
common_qids = set(qrel_df['qid']).intersection(set(query_df['qid']))
qrel_df = qrel_df[qrel_df['qid'].isin(common_qids)]
query_df = query_df[query_df['qid'].isin(common_qids)]

json_d = {
  "query_id": 1,
  "query": "!awnmower tires without rims",
  "positive_passages": [
    {
      "docid": 1049092,
      "title": "2 Pack 10-Inch Tires and Wheels 4.10/3.50-4 Replacement Utility Tires for Dolly, Hand Truck, Gorilla Cart, Generator, Lawn Mower, Garden Wagon With 5/8-Inch Axle Borehole and Double Sealed Bearings",
      "text": "Product Description Read more Read more All-purpose Utility The Ram-Pro’s ten inch, ready to install Air Tire Wheels is the best replacement for your hand truck tires and wheels. It offers easy installation and optimum effort reducing performance. The only effort you need to make is to remove the old tires before sliding in the new ones and closing them with clips, nuts or cotter pins. That’s a perfectly installed and proper functioning hand truck without any hesitation or frustration. Superior Quality The high-quality heavy duty rubber will last very long. The air stem is on the outside so you can easily inflate the tires if needed. The double sealed bearings will evenly distribute the load on your vehicle and help to balance the loud noise evenly thereby making you push less and easier. The reduction in noise levels further helps to reduce stress levels. High End Specification The Ram-Pro Air Tires size is ten inches high and three inches wide with a hole diameter of 5/8 inches. The hub depth of the tires is 1-3/4 inches with double sealed bearings. With such high-end specifications, it can withstand a load capacity of 300 lbs. (approximately 136 Kgs). The pressure per square inch rating is 30 per tire. The tube type is a 2 ply 4.10/3.5-4. These air-filled tires are designed with raised grips to make your drive the smoothest possible one. Numerous Applications The air tires can be used for hand trucks, lawn mowers, yard wagons, air compressors, power washers, child's wagons, shopping carts, wood chippers, snow blowers, dollies, go karts, golf carts, tricycles, and much more. This assembly has proven to be ideal for air compressors, dollies, generators, pressure washers and various other auxiliary utility equipment. Read more Read more "
    }
  ],
  "negative_passages": [
    {
      "docid": 689593,
      "title": "NEIKO 20601A 14.5” Steel Tire Spoons Tool Set, Tire Tools Include 3 Piece Tire Spoons, 3 Piece Rim Protector, Valve Tool, 6 Piece Valve Cores, Motorcycle Tire Changer, Dirt Bike Tire Levers",
      "text": "Product Description 20601A Neiko Tire Spoons Rim Protector and Valve Tool Set is the perfect tool for any mechanics, homeowners, or DIY-ers! Whether you’re working on a car or motorcycle, this complete package of tools is great for removing damaged tires. The set not only includes one, but three flexible rim guards to prevent any damage to the rims while using the tire spoons. As an added BONUS, the set includes 6 Brass Schrader Valve Cores for convenience. Product Features: - Tire spoons are constructed of high quality Hardened Steel-Iron for ultimate durability/strength - Chrome polish finish that protects the tool from rust/corrosion and allows for hassle-free cleaning - Curved tips allow for easy insertion to any and all applications - Extra long length lever at 14.5 inches to easily prop out damaged tires without using extra force - Ergonomic, contoured, rubberized groove handle provides a natural non-slip grip while reducing hand fatigue and maximizing leverage power - 4 point valve tool with mouth side for attachments of various bits, deflation needle side that can be used to deflate the needle or remove stuck rocks in tire grooves, valve core side for quick removal and installation of valve core stems, and the screw thread side to easily clean the threads of the wheel stems Product Description 20601A Neiko Tire Spoons Rim Protector and Valve Tool Set is the perfect tool for any mechanics, homeowners, or DIY-ers! Whether you’re working on a car or motorcycle, this complete package of tools is great for removing damaged tires. The set not only includes one, but three flexible rim guards to prevent any damage to the rims while using the tire spoons. As an added BONUS, the set includes 6 Brass Schrader Valve Cores for convenience. Product Features: - Tire spoons are constructed of high quality Hardened Steel-Iron for ultimate durability/strength - Chrome polish finish that protects the tool from rust/corrosion and allows for hassle-free cleaning - Curved tips allow for easy insertion to any and all applications - Extra long length lever at 14.5 inches to easily prop out damaged tires without using extra force - Ergonomic, contoured, rubberized groove handle provides a natural non-slip grip while reducing hand fatigue and maximizing leverage power - 4 point valve tool with mouth side for attachments of various bits, deflation needle side that can be used to deflate the needle or remove stuck rocks in tire grooves, valve core side for quick removal and installation of valve core stems, and the screw thread side to easily clean the threads of the wheel stemsFrom the manufacturer Read more Premium Tire Changing Tool Kit Quality Construction The hardened steel iron construction through out the lever shaft to the curved tips, yield the ultimate in strength and durability that gives this tool set supreme effectiveness when tire changing. What's Included: 3 Piece - Hardend Steel-Iron Tire Spoon Levers 3 Piece - Rim Protectors 1 Piece - 4-Way Valve Tool 6 Piece - Brass Schrader Valve Cores Read more Rust Resistant With a polished chrome finish, you don't need to worry about your tire spoons getting rust on them. Longer Reach At 14.5 inches long, you can gain the leverage and utilize the force needed on your tires with ease! Ergonomics The rubber grip is ergonomically designed for comfort and control, and allows you to maximize your leverage. 4-Way Valve Tool and 6 Valve Cores Included With the 4-Way valve tool, you have a clean thread, valve core remover, deflation needle and tapping mouth, all-in-one, in addition to 6 brass valve cores all included. Read more Read more "
    },
    {
      "docid": 1117073,
      "title": "Kenda Schwinn Tire",
      "text": "Product Description The Kenda K23 S-6 tire is a great replacement tire for older 3-speed Schwinn bicycles. 26\" x 1 3/8\" x 1 1/4\" (37-597 ISO) size. This is an obscure size found on some Schwinn bicycles from the 1960's and 70's. This tire is NOT compatible with 26\" x 1 3/8\" (590 ISO) rims, nor with 26\" (559 ISO) mountain bike rims. Match the ISO measurement on the sidewall of your current tire for proper compatability. Product Description The Kenda K23 S-6 tire is a great replacement tire for older 3-speed Schwinn bicycles. 26\" x 1 3/8\" x 1 1/4\" (37-597 ISO) size. This is an obscure size found on some Schwinn bicycles from the 1960's and 70's. This tire is NOT compatible with 26\" x 1 3/8\" (590 ISO) rims, nor with 26\" (559 ISO) mountain bike rims. Match the ISO measurement on the sidewall of your current tire for proper compatability.From the manufacturer KENDA. DESIGNED FOR YOUR JOURNEY On the road, on the trail or on the racetrack, you can count on Kenda quality. Our tires are engineered for performance and value across a wide range of interests and applications. See why Kenda is the right choice. It's your move. "
    }
  ]
}

def build_baseline():
    query = 'black running shoe for men nike'
    collection['text'] = collection['title'] + " " + collection['description']
    corpus = collection['text'].tolist()
    # Preprocess the corpus
    print('Preprocess the Product Collection')
    start = time.time()

    # tokenized_corpus = [preprocessor.preprocess(doc) for doc in corpus]

    # with open("../data/train/tokenized_corpus.json", "w") as outfile:
    #    json.dump(tokenized_corpus, outfile)

    with open("../data/train/tokenized_corpus.json", 'r') as openfile:
        tokenized_corpus = json.load(openfile)

    print(f'Preprocessing finished in {time.time() - start} seconds \n')

    # Define the parameter grid

    param_grid = {
        'k1': [1.2, 1.5, 1.8],
        'b': [0.6, 0.75, 0.9],
        'epsilon': [0.1, 0.25, 0.5]
    }


    # Grid search
    print(f'Start the Hyperparamter Tuning')
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=params['k1'], b=params['b'], epsilon=params['epsilon'])
        score = evaluate_bm25(query_df.sample(n=10, random_state=119), qrel_df, bm25, collection)
        if score > best_score:
            best_score = score
            best_params = params

    print("Best Score:", best_score)
    print("Best Parameters:", best_params)
    print(f'Took {time.time() - start} seconds \n')


    # build and save best BM25 model
    start = time.time()
    bm25 = BM25Okapi(tokenized_corpus, tokenizer=None, k1=best_params['k1'], b=best_params['b'], epsilon=best_params['epsilon'])
    print(f'Best BM25 initialized in {time.time() - start} seconds')
    tokenized_query = preprocessor.preprocess(query)
    start = time.time()
    print(f'Evaluate BM25 Model...')
    print(evaluate_bm25(query_df.sample(n=10, random_state=119), qrel_df, bm25, collection))
    print(f'Took {time.time() - start} seconds')

    bm25.save_model('../bm25_baseline.pkl')
    print('saved successfully')


if __name__ == "__main__":

    print('run main')
    data = read_jsonl_file("../data/train/train.jsonl")[:100]
    generator = Llama3QueryGenerator(data)
    """
    generator = BartQueryGenerator(data)
    generator.train(num_train_epochs=1)
    generator.save_model('../models/saved_model', '../models/saved_tokenizer')
    """
    test_description = "Mini portable 60% compact layout: MK-Box is a 68 keys mechanical keyboard have cute small size, separate arrow keys and all your F-keys you need, can use it for gaming or work while saving space." \
                       "Mechanical red switch: characterized for being linear and smoother, slight key sound has no paragraph sense with minimal resistance, but fast action without a tactile bump feel which makes it easier to tap the keyboard."
    query = generator.generate_query(test_description)
    print(query)





