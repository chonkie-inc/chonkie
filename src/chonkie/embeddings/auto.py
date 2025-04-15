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
    def get_embeddings(
        cls, model: Union[str, BaseEmbeddings, Any], **kwargs
    ) -> BaseEmbeddings:
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
            # Try to find matching implementation via registry
            try:
                embeddings_cls = EmbeddingsRegistry.match(model)
                if embeddings_cls and embeddings_cls.is_available():
                    try:
                        return embeddings_cls(model, **kwargs)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to load {embeddings_cls.__name__}: {e}. Falling back to default {embeddings_cls} model."
                        )
                        return embeddings_cls(**kwargs)
            except Exception as error:
                warnings.warn(
                    f"Failed to load embeddings via registry: {error}. Falling back to SentenceTransformerEmbeddings."
                )
                # Fall back to SentenceTransformerEmbeddings if no matching implementation is found
                from .sentence_transformer import SentenceTransformerEmbeddings

                try:
                    return SentenceTransformerEmbeddings(model, **kwargs)
                except Exception as e:
                    # try with ollama - as any gguf model can be loaded through ollama
                    from .ollama import OllamaEmbeddings
                    try:
                        return OllamaEmbeddings(model, **kwargs)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to load embeddings via SentenceTransformerEmbeddings & Ollama: {e}"
                        )
        else:
            # get the wrapped embeddings instance
            try:
                return EmbeddingsRegistry.wrap(model, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to wrap embeddings instance: {e}")
