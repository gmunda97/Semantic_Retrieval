from unittest.mock import patch
import numpy as np
import pandas as pd
import faiss
from semantic_search import SemanticSearch


sample_embeddings = np.random.rand(100, 384).astype('float32')
faiss.normalize_L2(sample_embeddings)
sample_texts = pd.read_csv("./datasets/papers.csv")["title"][:100].tolist()
sample_links = pd.read_csv("./datasets/papers.csv")["link"][:100].tolist()


class TestSemanticSearch:

    @classmethod
    def setup_class(cls):
        cls.semantic_search_instance = SemanticSearch(
            sample_embeddings,
            index_type='flatIP',
            cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

    def test_create_query(self):
        query = "sample query"
        query_embedding = self.semantic_search_instance.create_query(query)

        assert isinstance(query_embedding, np.ndarray)
        assert query_embedding.shape == (1, 384)

    def test_retrieve_documents(self):
        query = "sample query"
        number_of_documents = 4

        assert len(sample_texts) >= number_of_documents
        assert len(sample_links) >= number_of_documents

        with patch.object(self.semantic_search_instance, 'cross_encoder', None):
            results = self.semantic_search_instance.retrieve_documents(
                query,
                sample_texts[:number_of_documents],
                sample_links[:number_of_documents],
                number_of_documents=number_of_documents
            )

        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 4

    def test_rerank_documents(self):
        query = "sample query"
        results = [
            (0, "Document 1", 0.5, "Link 1"),
            (1, "Document 2", 0.4, "Link 2"),
            (2, "Document 3", 0.6, "Link 3"),
            (3, "Document 4", 0.2, "Link 4")
        ]
        reranked_results = self.semantic_search_instance.rerank_documents(query, results)

        assert isinstance(reranked_results, list)
        for reranked_result in reranked_results:
            assert isinstance(reranked_result, tuple)
            assert len(reranked_result) == 4
