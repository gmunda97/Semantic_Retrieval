import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class SemanticSearch():
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    INDEX_TYPES = {
        "flatL2": faiss.IndexFlatL2,
        "flatIP": faiss.IndexFlatIP,
    }

    def __init__(self, embeddings, model_name=None, index_type=None, index_file=None, cross_encoder_name=None):
        model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(model_name)
        if index_file is not None:
            self.index = self.load_index_from_file(index_file)
        else:
            self.index = self.create_index(embeddings, index_type)
        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name is not None else None

    def create_index(self, embeddings, index_type):
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

    def retrieve_documents(self, query, text_data, link_data, number_of_k=10, similarity_threshold=0.3):
        k = min(number_of_k, self.index.ntotal)
        query_vector = self.create_query(query)
        D, I = self.index.search(query_vector, k)
        # sorting the documents based on similarity and adding the links
        results = [(i, text_data[i], d, link_data[i]) 
                   for d, i in zip(D[0], I[0]) if d > similarity_threshold]
        
        if self.cross_encoder is not None:
            results = self.rerank_documents(query, results)
        
        return results
    
    def rerank_documents(self, query, results):
        cross_encoder = self.cross_encoder
        query_doc_pairs = [(query, text) for _, text, _, _ in results] 
        scores = cross_encoder.predict(query_doc_pairs)
        softmax_scores = np.exp(scores) / np.sum(np.exp(scores))
        reranked_results = [(i, text, score, link) for (i, text, _, link), 
                    score in zip(results, softmax_scores)]

        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)

        return reranked_results
    
    def save_index_to_file(self, index_file):
        faiss.write_index(self.index, index_file)
    
