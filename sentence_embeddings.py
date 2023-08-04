import pickle
import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

@dataclass
class ColumnNames:
    title: str = "title"
    link: str = "link"


class SentenceEmbeddings():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):
        documents = documents.to_list()
        embeddings = self.model.encode(documents)
        self.save_embeddings(embeddings)
        return embeddings

    def save_embeddings(self, embeddings, filename="embeddings_papers.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentence embeddings with SentenceTransformers.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset file')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"Dataset file does not exist: {dataset_path}")
        exit()
    
    model = SentenceEmbeddings('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    df = pd.read_csv(dataset_path)
    column_names = ColumnNames()
    documents = df[column_names.title]
    embeddings = model.generate_embeddings(documents)

    model.save_embeddings(embeddings)