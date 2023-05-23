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
    search = SemanticSearch(embeddings=embeddings, index_type="FlatIP", index_file=index, cross_encoder=cross_encoder)

    ground_truth_scores = [3, 2, 4, 1, 3, 2, 3, 4, 2, 1]

    query = "neural proabilistic language model"
    results = search.retrieve_documents(query, text_data=df["title"], link_data=df["link"])

    # Extract the relevance scores from the retrieved documents
    retrieved_scores = [score for _, _, score, _ in results]

    # Calculate NDCG score
    ndcg = ndcg_score([ground_truth_scores], [retrieved_scores])
    print(f"NDCG Score: {ndcg:.6f}")
