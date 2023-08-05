import argparse
import os
import pickle
import pandas as pd
from semantic_search import SemanticSearch
from sentence_embeddings import ColumnNames

'''
Main file to run the semantic search
'''

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

    cross_encoder_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

    search = SemanticSearch(embeddings=embeddings, 
                            index_type=args.index_type, 
                            index_file=args.index_path, 
                            cross_encoder_name=cross_encoder_name)
    
    column_names = ColumnNames()
    #search.load_index_from_file(args.index_file)
    #search.save_index_to_file("index_papers.index")
    df = pd.read_csv(dataset_path)

    while True:
        query = input('Input your query here (press "q" to quit): ')
        if query == "q":
            break
        
        results = search.retrieve_documents(query, 
                                            text_data=df[column_names.title], 
                                            link_data=df[column_names.link])

        for i, doc, score, link in results:
            print(f"Document {i} (score: {score:.4f}): {doc}")
            print(f"Link: {link} \n")