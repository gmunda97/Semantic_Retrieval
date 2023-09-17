'''
A simple script to evaluate the results of the semantic search
using the NDCG score on 5 queries.
'''

from typing import List
import pickle
import pandas as pd
from sklearn.metrics import ndcg_score

from semantic_search import SemanticSearch
from sentence_embeddings import ColumnNames


class SemanticSearchEvaluation:
    def __init__(
            self,
            dataset_path: str,
            embeddings_path: str,
            index_path: str,
            cross_encoder_name: str
        ):

        self.df = pd.read_csv(dataset_path)
        self.column_names = ColumnNames()

        with open(embeddings_path, 'rb') as f:
            self.embeddings = pickle.load(f)

        self.search = self.initialize_search(index_path, cross_encoder_name)

    def initialize_search(self, index_path: str, cross_encoder_name: str) -> SemanticSearch:
        search = SemanticSearch(embeddings=self.embeddings,
                                index_type="FlatIP",
                                index_file=index_path,
                                cross_encoder_name=cross_encoder_name)
        return search

    def evaluate_queries(self, queries: List[str], ground_truth_scores: List[List[int]]) -> None:
        for query, ground_truths in zip(queries, ground_truth_scores):
            results = self.search.retrieve_documents(query,
                                                     text_data=self.df[self.column_names.title],
                                                     link_data=self.df[self.column_names.link])
            retrieved_scores = [score for _, _, score, _ in results]
            ndcg = ndcg_score([ground_truths], [retrieved_scores])

            print(f"NDCG Score for '{query}': {ndcg:.6f}\n")


if __name__ == "__main__":
    DATASET_PATH = "datasets/papers.csv"
    EMBEDDINGS_PATH = "embeddings/embeddings_papers.pkl"
    INDEX_PATH = "index/index_papers.index"

    CROSS_ENCODER_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    ground_truth_scores = [
        [4, 4, 1, 4, 2, 3, 3, 2, 1, 1], # query 1
        [4, 3, 4, 4, 1, 1, 1, 0, 0, 0], # query 2
        [3, 4, 3, 2, 1, 4, 3, 3, 0, 0], # ...
        [0, 3, 2, 0, 3, 0, 4, 4, 3, 0],
        [4, 2, 2, 0, 2, 0, 1, 1, 0, 0]
    ]

    queries = [
        "neural proabilistic language model",
        "word embeddings for similarity search",
        "explainability of neural networks",
        "deep learning for computer vision",
        "quantum machine learning"
    ]

    evaluator = SemanticSearchEvaluation(DATASET_PATH,
                                         EMBEDDINGS_PATH,
                                         INDEX_PATH,
                                         CROSS_ENCODER_NAME)
    evaluator.evaluate_queries(queries, ground_truth_scores)
