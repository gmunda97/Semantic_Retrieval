from unittest.mock import patch
import numpy as np
import pandas as pd
import faiss
from ..scripts.semantic_search import SemanticSearch


sample_embeddings = np.random.rand(100, 384).astype('float32')
faiss.normalize_L2(sample_embeddings)
sample_texts = pd.read_csv(".././datasets/papers.csv")["title"][:100].tolist()
sample_links = pd.read_csv(".././datasets/papers.csv")["link"][:100].tolist()


class TestSemanticSearch:

    @classmethod
    def setup_class(cls):
        cls.semantic_search_instance = SemanticSearch(
            sample_embeddings,
            index_type='flatIP',
            cross_encoder_name='cross-encoder/ms-marco-MiniLM-L-6-v2'
        )

    def test_create_index_flatL2(self):
        sample_embeddings = np.random.rand(100, 384).astype('float32')
        index_type = 'flatL2'
        index = self.semantic_search_instance.create_index(sample_embeddings, index_type)
        assert isinstance(index, faiss.IndexFlatL2)

    def test_create_index_flatIP(self):
        sample_embeddings = np.random.rand(100, 384).astype('float32')
        index_type = 'flatIP'
        index = self.semantic_search_instance.create_index(sample_embeddings, index_type)
        assert isinstance(index, faiss.IndexFlatIP)

    @patch('faiss.read_index', autospec=True)
    def test_load_index_from_file(self, mock_read_index):
        index_file = "sample_index.index"
        mock_index = faiss.IndexFlatIP(384)
        mock_read_index.return_value = mock_index

        loaded_index = self.semantic_search_instance.load_index_from_file(index_file)
        # Check if faiss.read_index was called with the correct file path
        mock_read_index.assert_called_once_with(index_file)
        assert isinstance(loaded_index, faiss.IndexFlatIP)

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

    @patch("faiss.write_index", autospec=True)
    def test_save_index_to_file(self, mock_faiss_write_index):
        index_file = "sample_index.index"
        mock_faiss_write_index.return_value = None
        self.semantic_search_instance.save_index_to_file(index_file)

        # Check if faiss.write_index was called with the correct arguments
        mock_faiss_write_index.assert_called_once_with(self.semantic_search_instance.index, index_file)
