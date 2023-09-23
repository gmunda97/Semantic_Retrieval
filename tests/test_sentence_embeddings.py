import os
import pickle
import pytest
import pandas as pd
import numpy as np
from unittest.mock import mock_open, patch
from ..scripts.sentence_embeddings import SentenceEmbeddings

dataset = pd.read_csv(".././datasets/papers.csv")
documents = dataset["title"][:50]

class TestSentenceEmbeddings:

    @classmethod
    def setup_class(cls):
        cls.sentence_embeddings_instance = SentenceEmbeddings(
            model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
        )

    def test_generate_embeddings(self):
        embeddings = self.sentence_embeddings_instance.generate_embeddings(documents)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == len(documents)
        assert isinstance(embeddings[0], np.ndarray)
        assert np.isscalar(embeddings[0][0])

    @pytest.fixture
    def tmp_dir(self, tmpdir):
        # Temporary directory for testing
        return tmpdir.mkdir("test_data")

    def test_save_embeddings(self, tmp_dir):
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        file_path = os.path.join(tmp_dir, "test_embeddings.pkl")

        self.sentence_embeddings_instance.save_embeddings(embeddings, file_path)

        # Check if file exists
        assert os.path.isfile(file_path)

        with open(file_path, 'rb') as f:
            loaded_embeddings = pickle.load(f)

        assert loaded_embeddings == embeddings
