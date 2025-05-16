"""Google Vertex AI embeddings implementation."""

import importlib.util as importutil
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from .base import BaseEmbeddings

if TYPE_CHECKING:
    try:
        import numpy as np
    except ImportError:
        np = Any  # type: ignore


class VertexAIEmbeddings(BaseEmbeddings):
    """Google Vertex AI embeddings implementation using Google Cloud API.
    
    Args:
        model: The name of the embedding model to use.
        project_id: The Google Cloud project ID. If not provided, will attempt to determine from the environment.
        location: The region to use. Defaults to "us-central1".
        api_key: The API key to use. If not provided, will look for GOOGLE_API_KEY environment variable.
        credentials_path: Path to service account credentials JSON file.
        batch_size: Maximum number of texts to embed in one API call.
        show_warnings: Whether to show warnings about token usage.
    """

    AVAILABLE_MODELS = {
        "textembedding-gecko@001": 768,  # Gecko, first-gen model
        "textembedding-gecko@latest": 768,  # Latest Gecko version
        "textembedding-gecko-multilingual@latest": 768,  # Multilingual Gecko
        "textembedding-gecko-multilingual@001": 768,  # Multilingual Gecko first-gen
        "text-embedding-004": 768,  # PaLM 3 
        "text-multilingual-embedding-002": 768,  # Multilingual PaLM 3
    }

    DEFAULT_MODEL = "text-embedding-004"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        api_key: Optional[str] = None,
        credentials_path: Optional[str] = None,
        batch_size: int = 5,
        show_warnings: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Initialize the Google Vertex AI embeddings client.

        Args:
            model: Name of the embedding model to use.
            project_id: Google Cloud project ID.
            location: Google Cloud region.
            api_key: API key for authentication (or set GOOGLE_API_KEY env var).
            credentials_path: Path to service account credentials JSON file.
            batch_size: Maximum number of texts to embed in one API call.
            show_warnings: Whether to show warnings about token usage.
            **kwargs: Additional keyword arguments to pass to the Vertex AI client.
        """
        super().__init__()

        # Lazy import dependencies
        self._import_dependencies()

        # Set up model parameters
        self.model = model
        self.location = location
        self._batch_size = batch_size
        self._show_warnings = show_warnings
        self.kwargs = kwargs

        # Check if the model is supported
        if model not in self.AVAILABLE_MODELS and self._show_warnings:
            warnings.warn(
                f"Model {model!r} not in known models list: {list(self.AVAILABLE_MODELS.keys())}. "
                "Proceeding anyway but dimension may not be correctly set."
            )
            # Default to a reasonable dimension for unknown models
            self._dimension = 768
        else:
            self._dimension = self.AVAILABLE_MODELS.get(model, 768)

        # Set up authentication
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Initialize the Vertex AI client
        if api_key:
            # Use API key authentication
            aiplatform.init(
                project=project_id,
                location=location,
                api_key=api_key,
            )
        elif credentials_path:
            # Use service account credentials
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            aiplatform.init(
                project=project_id,
                location=location,
                credentials=credentials,
            )
        else:
            # Use application default credentials
            aiplatform.init(
                project=project_id,
                location=location,
            )

        # Create endpoint resource
        self._endpoint = f"projects/{project_id or aiplatform.initializer.global_config.project}/locations/{location}/publishers/google/models/{model}"

    def embed(self, text: str) -> "np.ndarray":
        """Get embeddings for a single text.
        
        Args:
            text: The input text to embed.
            
        Returns:
            A NumPy array of the embedding vector.
        """
        try:
            embeddings = self.embed_batch([text])
            return embeddings[0]
        except Exception as e:
            raise RuntimeError(f"Google Vertex AI embeddings API error: {e}") from e

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Get embeddings for a batch of texts.
        
        Args:
            texts: List of input strings to embed.
            
        Returns:
            List of NumPy arrays representing embedding vectors.
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to avoid hitting API limits
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            
            try:
                # Create prediction request payload
                instances = [{"content": text} for text in batch]
                
                # Use the Endpoint.predict method
                endpoint = aiplatform.Endpoint(self._endpoint)
                response = endpoint.predict(instances=instances)
                
                # Extract embeddings from the response
                embedding_values = []
                for prediction in response.predictions:
                    if "embeddings" in prediction:
                        # Format for text-embedding-004
                        values = prediction["embeddings"]["values"]
                        embedding_values.append(np.array(values, dtype=np.float32))
                    elif "embedding" in prediction:
                        # Format for textembedding-gecko
                        values = prediction["embedding"]
                        embedding_values.append(np.array(values, dtype=np.float32))
                    else:
                        raise ValueError(f"Unexpected response format: {prediction}")
                
                all_embeddings.extend(embedding_values)
                
            except Exception as e:
                # If batch fails, try one by one
                if len(batch) > 1 and self._show_warnings:
                    warnings.warn(f"Batch embedding failed: {str(e)}. Trying one by one.")
                    for text in batch:
                        try:
                            # Use single prediction
                            endpoint = aiplatform.Endpoint(self._endpoint)
                            response = endpoint.predict(instances=[{"content": text}])
                            
                            # Extract the embedding
                            prediction = response.predictions[0]
                            if "embeddings" in prediction:
                                values = prediction["embeddings"]["values"]
                                all_embeddings.append(np.array(values, dtype=np.float32))
                            elif "embedding" in prediction:
                                values = prediction["embedding"]
                                all_embeddings.append(np.array(values, dtype=np.float32))
                            else:
                                raise ValueError(f"Unexpected response format: {prediction}")
                        except Exception as inner_e:
                            raise RuntimeError(f"Failed to embed text: {inner_e}") from inner_e
                else:
                    raise RuntimeError(f"Google Vertex AI embeddings API error: {e}") from e

        return all_embeddings

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    def get_tokenizer_or_token_counter(self) -> Union[Any, callable]:
        """Return a function that counts tokens.
        
        Since Vertex AI doesn't expose its tokenizer directly, we don't have a 
        precise way to count tokens. In practice, this wouldn't cause issues as
        Vertex AI models handle truncation internally.
        
        Returns:
            A simple function that estimates tokens.
        """
        # Approximation: ~4 characters per token for English text
        return lambda text: len(text) // 4

    def is_available(self) -> bool:
        """Check if the required packages are available."""
        return (
            importutil.find_spec("google.cloud.aiplatform") is not None
            and importutil.find_spec("numpy") is not None
        )

    def _import_dependencies(self) -> None:
        """Lazy import dependencies if they are not already imported."""
        if self.is_available():
            global np, aiplatform
            import numpy as np
            from google.cloud import aiplatform
        else:
            raise ImportError(
                "One (or more) of the following packages is not available: "
                "google-cloud-aiplatform, numpy. "
                "Please install them via `pip install \"chonkie[vertexai]\"` "
                "or `pip install google-cloud-aiplatform numpy`"
            )

    def __repr__(self) -> str:
        """Return a string representation of the embeddings instance."""
        return f"VertexAIEmbeddings(model={self.model}, location={self.location})" 