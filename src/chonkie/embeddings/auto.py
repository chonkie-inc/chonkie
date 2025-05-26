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

        """
        # Load embeddings instance if already provided
        if isinstance(model, BaseEmbeddings):
            return model
        elif isinstance(model, str):
            # Initializing the embedding instance
            embeddings_instance = None

            # Check if the user passed in a provider alias
            if "://" in model:
                provider, model_name = model.split("://")
                embeddings_cls = EmbeddingsRegistry.get_provider(provider)
                if embeddings_cls:
                    try:
                        return embeddings_cls(model_name, **kwargs)  # type: ignore
                    except Exception as error:
                        raise ValueError(f"Failed to load {model} with {embeddings_cls.__name__}, with error: {error}")
                else:
                    raise ValueError(f"No provider found for {provider}. Please check the provider name and try again.")
            else:
                # Try to find matching implementation via registry
                embeddings_cls = EmbeddingsRegistry.match(model)
                if embeddings_cls:
                        try:
                            # Try instantiating with the model identifier
                            embeddings_instance = embeddings_cls(model, **kwargs)  # type: ignore
                        except Exception as error:
                            warnings.warn(
                                f"Failed to load {model} with {embeddings_cls.__name__}: {error}\n"
                                f"Falling back to loading default provider model."
                            )
                            try:
                                # Try instantiating with the default provider model without the model identifier
                                embeddings_instance = embeddings_cls(**kwargs)
                            except Exception as error:
                                warnings.warn(
                                    f"Failed to load the default model for {embeddings_cls.__name__}: {error}\n"
                                    f"Falling back to SentenceTransformerEmbeddings."
                                )

            # If registry lookup and instantiation succeeded, return the instance
            if embeddings_instance:
                return embeddings_instance

            # If registry lookup and instantiation failed, return the default SentenceTransformerEmbeddings
            from .sentence_transformer import SentenceTransformerEmbeddings
            try:
                return SentenceTransformerEmbeddings(model, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to load embeddings via SentenceTransformerEmbeddings after registry/fallback failure: {e}")
        else:
            # get the wrapped embeddings instance
            try:
                return EmbeddingsRegistry.wrap(model, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to wrap embeddings instance: {e}")
