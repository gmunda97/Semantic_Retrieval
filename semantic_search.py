import torch
import nltk
import faiss
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer


class SentenceEmbeddings():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, documents):
        documents = documents.to_list()
        embeddings = self.model.encode(documents)
        return embeddings
    
class SemanticSearch(SentenceEmbeddings):
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'

    def __init__(self, documents, model_name=None):
        model_name = model_name or self.DEFAULT_MODEL_NAME
        super().__init__(model_name)
        self.documents = documents
        self.index = self.add_to_index(self.generate_embeddings(documents))

    def add_to_index(self, embeddings):
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
    #embedder = SentenceEmbeddings()
    df = pd.read_csv("New_DeepLearning_dataset.csv")
    documents = df["text"][:100]
    search = SemanticSearch(documents)
    query = "fine-tuning BERT"
    query_embedding = search.create_query(query)
    results = search.retrieve_query(query_embedding)
    print(results)
    #embeddings = embedder.generate_embeddings(documents)