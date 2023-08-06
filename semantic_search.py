import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Union, Optional


class SemanticSearch():
    DEFAULT_MODEL_NAME = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    INDEX_TYPES = {
        "flatL2": faiss.IndexFlatL2,
        "flatIP": faiss.IndexFlatIP,
    }

    def __init__(
            self, 
            embeddings: Union[list, faiss.Index], 
            model_name: Optional[str] = None,
            index_type: Optional[str] = None,
            index_file: Optional[str] = None, 
            cross_encoder_name: Optional[str] = None
    ) -> None:
        
        model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(model_name)
        if index_file is not None:
            self.index = self.load_index_from_file(index_file)
        else:
            self.index = self.create_index(embeddings, index_type)
        self.cross_encoder = CrossEncoder(cross_encoder_name) if cross_encoder_name is not None else None

    def create_index(self, embeddings: list, index_type: str) -> faiss.Index:
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
    
    def load_index_from_file(self, index_file: str) -> faiss.Index:
        index = faiss.read_index(index_file)

        return index

    def create_query(self, query: str) -> np.ndarray:
        query_embedding = self.model.encode([query])

        return query_embedding

    def retrieve_documents(
            self, 
            query: str, 
            text_data: list, 
            link_data: list, 
            number_of_documents: int = 10, 
            similarity_threshold: float = 0.3
    ) -> list:
        
        max_documents_to_retrieve = min(number_of_documents, self.index.ntotal)
        query_embedding = self.create_query(query)
        distances, documents_indices = self.index.search(query_embedding, max_documents_to_retrieve)

        # sorting the documents based on similarity and adding the links
        retrieved_documents = [
            (index, text_data[index], distance, link_data[index]) 
            for distance, index in zip(distances[0], documents_indices[0]) 
            if distance > similarity_threshold
        ]
        
        if self.cross_encoder is not None:
            retrieved_documents = self.rerank_documents(query, retrieved_documents)
        
        return retrieved_documents
    
    def rerank_documents(self, query: str, retrieved_documents: list) -> list:
        cross_encoder = self.cross_encoder
        query_doc_pairs = [(query, text) for _, text, _, _ in retrieved_documents]

        similarity_scores = cross_encoder.predict(query_doc_pairs)
        softmax_scores = np.exp(similarity_scores) / np.sum(np.exp(similarity_scores))

        reranked_results = [
            (index, text, score, link) 
            for (index, text, _, link), score in zip(retrieved_documents, softmax_scores)
        ]
        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)

        return reranked_results
    
    def save_index_to_file(self, index_file: str) -> None:
        faiss.write_index(self.index, index_file)
    
