import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SemanticSearch():
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    INDEX_TYPES = {
        "flatL2": faiss.IndexFlatL2,
        "flatIP": faiss.IndexFlatIP,
    }

    def __init__(self, embeddings, model_name=None, index_type=None):
        model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(model_name)
        self.index = self.load_index(embeddings, index_type)

    def load_index(self, embeddings, index_type):
        docs = embeddings
        d = embeddings.shape[1]

        if index_type == "flatL2":
            index = faiss.IndexFlatL2(d)
        elif index_type == "flatIP":
            index = faiss.IndexFlatIP(d) 
            faiss.normalize_L2(docs)
        else:
            raise ValueError(f"Invalid index type: {index_type}")
        
        index.add(docs.astype("float32"))
        index.ntotal
        return index

    def create_query(self, query):
        xq = self.model.encode([query])
        return xq

    def retrieve_query(self, embedding_vector, text_data, number_of_k=4):
        k = number_of_k
        D, I = self.index.search(embedding_vector, k)
        # sorting the documents based on similarity
        results = [(i, text_data[i], d) for d, i in zip(D[0], I[0])]
        results = sorted(results, key=lambda x: x[2], reverse=True)
        top_k_results = results[:k]
        return top_k_results
    

if __name__ == '__main__':
    # load the saved embeddings from file
    with open('embeddings3.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    search = SemanticSearch(embeddings=embeddings, index_type="flatIP")
    df = pd.read_csv("datasets/New_DeepLearning_dataset.csv")

    query = 'fine-tuning BERT'
    embedding_vector = search.create_query(query)
    results = search.retrieve_query(embedding_vector, text_data=df["text"])
    for i, doc, score in results:
        print(f"Document {i} (score: {score:.4f}): {doc}")