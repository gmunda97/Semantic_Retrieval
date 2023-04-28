import pickle
import faiss
import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer


class SemanticSearch():
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    INDEX_TYPES = {
        "flatL2": faiss.IndexFlatL2,
        "flatIP": faiss.IndexFlatIP,
    }

    def __init__(self, embeddings, model_name=None, index_type=None, index_file=None):
        model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(model_name)
        if index_file is not None:
            self.index = self.load_index_from_file(index_file)
        else:
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
    
    def load_index_from_file(self, index_file):
        index = faiss.read_index(index_file)
        return index

    def create_query(self, query):
        xq = self.model.encode([query])
        return xq

    def retrieve_query(self, embedding_vector, text_data, link_data, number_of_k=10):
        k = number_of_k
        D, I = self.index.search(embedding_vector, k)
        # sorting the documents based on similarity and adding the links
        results = [(i, text_data[i], D[0][j], link_data[i]) for j, i in enumerate(I[0])]
        results = sorted(results, key=lambda x: x[2], reverse=True)
        top_k_results = results[:k]
        return top_k_results 
    
    def save_index_to_file(self, index_file):
        faiss.write_index(self.index, index_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic search with SentenceTransformers and Faiss.')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset file')
    parser.add_argument('embeddings_path', type=str, help='Path to the embeddings file')
    parser.add_argument('--index_type', type=str, default='flatIP', help='Type of Faiss index')
    parser.add_argument('index_path', type=str, default=None, help='Path to the index file')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        print(f"Dataset file does not exist: {dataset_path}")
        exit()
    
    embeddings_path = args.embeddings_path
    if not os.path.exists(embeddings_path):
        print(f"Embeddings file does not exist: {embeddings_path}")
        exit()

    index_path = args.index_path
    if not os.path.exists(index_path):
        print(f"Index file does not exist: {index_path}")
        exit()

    #load the saved embeddings from file
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    search = SemanticSearch(embeddings=embeddings, index_type=args.index_type, index_file=args.index_path)
    #search.load_index_from_file(args.index_file)
    #search.save_index_to_file("index_papers.index")
    df = pd.read_csv(dataset_path)

    while True:
        query = input('Input your query here (press "q" to quit): ')
        if query == "q":
            break
        embedding_vector = search.create_query(query)
        results = search.retrieve_query(embedding_vector, text_data=df["title"], link_data=df["link"])
        for i, doc, score, link in results:
            print(f"Document {i} (score: {score:.4f}): {doc}")
            print(f"Link: {link} \n")