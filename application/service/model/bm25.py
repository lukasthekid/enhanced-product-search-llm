import math
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.tokenizer = tokenizer
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        doc_freqs = []
        nd = {}
        total_doc_length = 0

        for document in corpus:
            self.corpus_size += 1
            doc_length = len(document)
            self.doc_len.append(doc_length)
            total_doc_length += doc_length

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            doc_freqs.append(frequencies)

            for word in frequencies:
                if word in nd:
                    nd[word] += 1
                else:
                    nd[word] = 1

        self.avgdl = total_doc_length / self.corpus_size
        self.doc_freqs = doc_freqs
        return nd

    def _tokenize_corpus(self, corpus):
        with Pool(cpu_count()) as pool:
            tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        scores = self.get_scores(query)
        top_n_indices = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n_indices]

    def save_model(self, filepath):
        model_data = {
            'corpus_size': self.corpus_size,
            'avgdl': self.avgdl,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf,
            'doc_len': self.doc_len,
            'k1': getattr(self, 'k1', None),
            'b': getattr(self, 'b', None),
            'epsilon': getattr(self, 'epsilon', None)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        idf_sum = 0
        negative_idfs = []

        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)

        average_idf = idf_sum / len(self.idf)
        epsilon_idf = self.epsilon * average_idf
        for word in negative_idfs:
            self.idf[word] = epsilon_idf

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        for q in query:
            for i, doc in enumerate(self.doc_freqs):
                freq = doc.get(q, 0)
                if freq:
                    score = (self.idf.get(q, 0) * freq * (self.k1 + 1) /
                             (freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)))
                    scores[i] += score
        return scores.tolist()
