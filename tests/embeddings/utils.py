from unittest.mock import MagicMock


def make_mock_catsu_client(dimension: int = 1024, model_name: str = "default-model"):
    """Create a mock Catsu client for testing."""
    import numpy as np

    mock_client = MagicMock()

    mock_embed_response = MagicMock()
    mock_embed_response.to_numpy.return_value = np.random.rand(1, dimension).astype(np.float32)
    mock_client.embed.return_value = mock_embed_response

    mock_model_info = MagicMock()
    mock_model_info.name = model_name
    mock_model_info.dimensions = dimension
    mock_client.list_models.return_value = [mock_model_info]

    mock_tokenize_response = MagicMock()
    mock_tokenize_response.token_count = 10
    mock_client.tokenize.return_value = mock_tokenize_response

    return mock_client
