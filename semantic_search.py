import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SemanticSearch():
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'

    def __init__(self, embeddings, model_name=None):
        model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(model_name)
        self.index = self.load_index(embeddings)

    def load_index(self, embeddings):
        docs = embeddings
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(docs.astype("float32"))
        index.ntotal
        return index

    def create_query(self, query):
        xq = self.model.encode([query])
        return xq

    def retrieve_query(self, embedding_vector, number_of_k=4):
        k = number_of_k
        D, I = self.index.search(embedding_vector, k)
        results = [f"{i}: {df['text'][i]}" for i in I[0]]
        return results
    

if __name__ == '__main__':
    # load the saved embeddings from file
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    search = SemanticSearch(embeddings=embeddings)
    df = pd.read_csv("New_DeepLearning_dataset.csv")

    query = 'fine-tuning BERT'
    embedding_vector = search.create_query(query)
    results = search.retrieve_query(embedding_vector)
    for doc in results:
        print(doc)