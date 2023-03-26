import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class SentenceEmbeddings():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):
        documents = documents.to_list()
        embeddings = self.model.encode(documents)
        self.save_embeddings(embeddings)
        return embeddings

    def save_embeddings(self, embeddings, filename='embeddings3.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)


if __name__ == '__main__':
    model = SentenceEmbeddings('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    df = pd.read_csv("datasets/New_DeepLearning_dataset.csv")
    documents = df["text"] 
    embeddings = model.generate_embeddings(documents)

    model.save_embeddings(embeddings)