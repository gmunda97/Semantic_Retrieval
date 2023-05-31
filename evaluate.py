import pickle
import pandas as pd
from sklearn.metrics import ndcg_score
from semantic_search import SemanticSearch
from sentence_transformers import CrossEncoder


if __name__ == "__main__":
    df = pd.read_csv("datasets/papers.csv")

    with open("embeddings/embeddings_papers.pkl", 'rb') as f:
        embeddings = pickle.load(f)

    index = "index/index_papers.index"

    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    search = SemanticSearch(embeddings=embeddings, 
                            index_type="FlatIP", 
                            index_file=index, 
                            cross_encoder=cross_encoder)

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
    
    for query, ground_truths in zip(queries, ground_truth_scores):
        results = search.retrieve_documents(query, text_data=df["title"], link_data=df["link"])
        retrieved_scores = [score for _, _, score, _ in results]
        ndcg = ndcg_score([ground_truths], [retrieved_scores])
        print(f"NDCG Score for '{query}': {ndcg:.6f}\n")
