"""AutoEmbeddings is a factory class for automatically loading embeddings."""

import warnings
from typing import Any, Union

from .base import BaseEmbeddings
from .registry import EmbeddingsRegistry


class AutoEmbeddings:
    """Factory class for automatically loading embeddings.

    This class provides a factory interface for loading embeddings based on an
    identifier string. It will try to find a matching embeddings implementation
    based on the identifier and load it with the provided arguments.


    Examples:
        # Get sentence transformers embeddings
        embeddings = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")

        # Get OpenAI embeddings
        embeddings = AutoEmbeddings.get_embeddings("openai://text-embedding-ada-002", api_key="...")

        # Get Anthropic embeddings
        embeddings = AutoEmbeddings.get_embeddings("anthropic://claude-v1", api_key="...")

         # Get Cohere embeddings
        embeddings = AutoEmbeddings.get_embeddings("cohere://embed-english-light-v3.0", api_key="...")
        
        # Get Ollama embeddings
        embeddings = AutoEmbeddings.get_embeddings("ollama://all-minilm")

    """

    @classmethod
    def get_embeddings(cls, model: Union[str, BaseEmbeddings, Any], **kwargs: Any) -> BaseEmbeddings:
        """Get embeddings instance based on identifier.

        Args:
            model: Identifier for the embeddings (name, path, URL, etc.)
            **kwargs: Additional arguments passed to the embeddings constructor

        Returns:
            Initialized embeddings instance

        Raises:
            ValueError: If no suitable embeddings implementation is found

        Examples:
            # Get sentence transformers embeddings
            embeddings = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")

            # Get OpenAI embeddings
            embeddings = AutoEmbeddings.get_embeddings("openai://text-embedding-ada-002", api_key="...")

            # Get Anthropic embeddings
            embeddings = AutoEmbeddings.get_embeddings("anthropic://claude-v1", api_key="...")

            # Get Cohere embeddings
            embeddings = AutoEmbeddings.get_embeddings("cohere://embed-english-light-v3.0", api_key="...")

            # Get Ollama embeddings
            embeddings = AutoEmbeddings.get_embeddings("ollama://all-minilm")

        """
        # Load embeddings instance if already provided
        if isinstance(model, BaseEmbeddings):
            return model
        elif isinstance(model, str):
            embeddings_instance = None
            try:
                # Try to find matching implementation via registry
                embeddings_cls = EmbeddingsRegistry.match(model)
                if embeddings_cls:
                    try:
                        # Try instantiating with the model identifier
                        embeddings_instance = embeddings_cls(model, **kwargs)  # type: ignore
                    except Exception as e:
                        warnings.warn(
                            f"Failed to load {embeddings_cls.__name__} with model identifier: {e}. Trying fallback constructor."
                        )
                        try:
                            # Try instantiating without the model identifier
                            embeddings_instance = embeddings_cls(**kwargs)
                        except Exception as e2:
                            warnings.warn(
                                f"Fallback constructor {embeddings_cls.__name__}(**kwargs) also failed: {e2}. Proceeding to SentenceTransformer fallback."
                            )
                            # If both fail, embeddings_instance remains None, proceed to fallback
                # If embeddings_cls is None, embeddings_instance remains None, proceed to fallback
            except Exception as error:
                warnings.warn(
                    f"Error during registry lookup/instantiation: {error}. Falling back to SentenceTransformerEmbeddings."
                )
                # If registry lookup itself fails, embeddings_instance remains None, proceed to fallback

            # If registry lookup and instantiation succeeded, return the instance
            if embeddings_instance:
                return embeddings_instance

            # Fallback to SentenceTransformerEmbeddings if registry failed or instantiation failed
            warnings.warn("Falling back to SentenceTransformerEmbeddings.")
            from .sentence_transformer import SentenceTransformerEmbeddings

            try:
                return SentenceTransformerEmbeddings(model, **kwargs)
            except Exception as e:
                # Try with ollama - as any gguf model can be loaded through ollama
                from .ollama import OllamaEmbeddings
                try:
                    return OllamaEmbeddings(model, **kwargs)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load embeddings via SentenceTransformerEmbeddings & OllamaEmbeddings after registry/fallback failure: {e}"
                    )
        else:
            # get the wrapped embeddings instance
            try:
                return EmbeddingsRegistry.wrap(model, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to wrap embeddings instance: {e}")
