import os
import pickle
from unittest.mock import mock_open, patch
import pytest
import pandas as pd
import numpy as np
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

    @patch('os.path.isfile', return_value=False)
    @patch('builtins.open', new_callable=mock_open, create=True)
    def test_save_embeddings(self, mock_open_file, tmp_dir):
        embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        file_path = os.path.join(tmp_dir, "test_embeddings.pkl")

        with patch('os.path.isfile', return_value=False):
            self.sentence_embeddings_instance.save_embeddings(embeddings, file_path)

        # Check if the file was opened in write binary mode ('wb')
        mock_open_file.assert_called_once_with(file_path, 'wb')
        # Check if the pickle.dump method was called
        mock_open_file().write.assert_called_once()

        _, args, _ = mock_open_file().write.mock_calls[0]
        loaded_embeddings = pickle.loads(args[0])

        assert loaded_embeddings == embeddings
