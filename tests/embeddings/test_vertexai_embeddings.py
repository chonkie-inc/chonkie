"""Tests for Google Vertex AI embeddings implementation."""

import importlib.util as importutil
import os
import unittest
from unittest import mock

import numpy as np
import pytest

from chonkie.embeddings import VertexAIEmbeddings


@pytest.mark.skipif(
    importutil.find_spec("google.cloud.aiplatform") is None,
    reason="google.cloud.aiplatform is not installed",
)
class TestVertexAIEmbeddings(unittest.TestCase):
    """Tests for the VertexAIEmbeddings class."""

    def setUp(self):
        """Set up test environment."""
        # Mock the aiplatform.init and Endpoint class
        self.aiplatform_init_patch = mock.patch("google.cloud.aiplatform.init")
        self.endpoint_patch = mock.patch("google.cloud.aiplatform.Endpoint")
        
        # Start the patches
        self.mock_aiplatform_init = self.aiplatform_init_patch.start()
        self.mock_endpoint_class = self.endpoint_patch.start()
        
        # Set up mock return values
        self.mock_endpoint = mock.MagicMock()
        self.mock_endpoint_class.return_value = self.mock_endpoint
        
        # Mock the predict response
        mock_response = mock.MagicMock()
        mock_response.predictions = [{"embeddings": {"values": [0.1] * 768}}]
        self.mock_endpoint.predict.return_value = mock_response
        
        # Create a global config mock for project_id fallback
        self.global_config_patch = mock.patch(
            "google.cloud.aiplatform.initializer.global_config",
            project="default-project"
        )
        self.mock_global_config = self.global_config_patch.start()
        
        # Create embeddings object
        self.embeddings = VertexAIEmbeddings(
            model="text-embedding-004", 
            project_id="test-project",
            location="us-central1"
        )

    def tearDown(self):
        """Clean up after tests."""
        # Stop the patches
        self.aiplatform_init_patch.stop()
        self.endpoint_patch.stop()
        self.global_config_patch.stop()

    def test_initialization(self):
        """Test that the embeddings can be initialized."""
        # Test with default parameters
        embeddings = VertexAIEmbeddings(project_id="test-project")
        
        assert embeddings.model == "text-embedding-004"
        assert embeddings.dimension == 768
        
        # Test with different model
        embeddings = VertexAIEmbeddings(
            model="textembedding-gecko@latest", 
            project_id="test-project"
        )
        assert embeddings.model == "textembedding-gecko@latest"
        assert embeddings.dimension == 768
        
        # Verify init was called
        self.mock_aiplatform_init.assert_called()

    def test_embed(self):
        """Test that embed method returns correct shape."""
        # Test embedding a single text
        embedding = self.embeddings.embed("Hello, world!")
        
        # Verify the embedding shape and type
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)
        assert embedding.dtype == np.float32
        
        # Verify the method was called correctly
        self.mock_endpoint_class.assert_called_once()
        self.mock_endpoint.predict.assert_called_once_with(instances=[{"content": "Hello, world!"}])

    def test_embed_batch(self):
        """Test that embed_batch method works correctly."""
        # Set up mock for batch embedding
        mock_response = mock.MagicMock()
        mock_response.predictions = [
            {"embeddings": {"values": [0.1] * 768}},
            {"embeddings": {"values": [0.2] * 768}},
        ]
        self.mock_endpoint.predict.return_value = mock_response
        
        # Test embedding a batch of texts
        embeddings = self.embeddings.embed_batch(["Hello", "World"])
        
        # Verify the embeddings shape and type
        assert len(embeddings) == 2
        assert isinstance(embeddings[0], np.ndarray)
        assert embeddings[0].shape == (768,)
        assert embeddings[1].shape == (768,)
        
        # Verify the batch call
        expected_instances = [{"content": "Hello"}, {"content": "World"}]
        self.mock_endpoint.predict.assert_called_with(instances=expected_instances)

    def test_batch_with_gecko_format(self):
        """Test embedding with textembedding-gecko response format."""
        # Set up mock for gecko format response
        mock_response = mock.MagicMock()
        mock_response.predictions = [
            {"embedding": [0.1] * 768},
            {"embedding": [0.2] * 768},
        ]
        self.mock_endpoint.predict.return_value = mock_response
        
        # Test embedding a batch of texts
        embeddings = self.embeddings.embed_batch(["Hello", "World"])
        
        # Verify we can handle both response formats
        assert len(embeddings) == 2
        assert embeddings[0].shape == (768,)

    def test_dimension_property(self):
        """Test dimension property."""
        assert self.embeddings.dimension == 768

    def test_repr(self):
        """Test string representation."""
        assert repr(self.embeddings) == "VertexAIEmbeddings(model=text-embedding-004, location=us-central1)"

    def test_tokenizer(self):
        """Test the token counter approximation."""
        token_counter = self.embeddings.get_tokenizer_or_token_counter()
        
        # Test with simple text
        tokens = token_counter("Hello, world!")
        
        # Simple approximation: 13 chars / 4 = 3 tokens
        assert tokens == 3


if __name__ == "__main__":
    unittest.main() 