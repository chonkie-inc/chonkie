"""Registry for embedding implementations with pattern matching support."""

import re
from typing import Any, Dict, List, Optional, Pattern, Type, Union

from .base import BaseEmbeddings
from .cohere import CohereEmbeddings
from .gemini import GeminiEmbeddings
from .jina import JinaEmbeddings
from .model2vec import Model2VecEmbeddings
from .openai import OpenAIEmbeddings
from .sentence_transformer import SentenceTransformerEmbeddings
from .voyageai import VoyageAIEmbeddings


class EmbeddingsRegistry:
    """Registry for embedding implementations with pattern matching support."""

    # Create a registry for the model names, provider aliases, patterns, and supported types
    model_registry: Dict[str, Type[BaseEmbeddings]] = {}
    provider_registry: Dict[str, Type[BaseEmbeddings]] = {}
    pattern_registry: Dict[Pattern, Type[BaseEmbeddings]] = {}
    type_registry: Dict[str, Type[BaseEmbeddings]] = {}

    @classmethod
    def register_model(
        cls,
        name: str,
        embedding_cls: Type[BaseEmbeddings]
    ) -> None:
        """Register a new embeddings implementation.

        Args:
            name: Unique identifier for this implementation
            embedding_cls: The embeddings class to register
            pattern: Optional regex pattern string or compiled pattern
            supported_types: Optional list of types that the embeddings class supports

        """
        if not issubclass(embedding_cls, BaseEmbeddings):
            raise ValueError(f"{embedding_cls} must be a subclass of BaseEmbeddings")

        cls.model_registry[name] = embedding_cls

    @classmethod
    def register_provider(
        cls,
        alias: str,
        embeddings_cls: Type[BaseEmbeddings],
    ) -> None:
        """Register a new provider.

        Args:
            alias: Unique identifier for this provider
            embeddings_cls: The embeddings class to register

        """
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        cls.provider_registry[alias] = embeddings_cls
    
    @classmethod
    def register_pattern(
        cls,
        pattern: str,
        embeddings_cls: Type[BaseEmbeddings]
    ) -> None:
        """Register a new pattern."""
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        compiled_pattern = re.compile(pattern)
        cls.pattern_registry[compiled_pattern] = embeddings_cls

    @classmethod
    def register_types(
        cls,
        types: Union[str, List[str]],
        embeddings_cls: Type[BaseEmbeddings]
    ) -> None:
        """Register a new type."""
        if not issubclass(embeddings_cls, BaseEmbeddings):
            raise ValueError(f"{embeddings_cls} must be a subclass of BaseEmbeddings")

        if isinstance(types, str):
            cls.type_registry[types] = embeddings_cls
        elif isinstance(types, list):
            for type in types:
                cls.type_registry[type] = embeddings_cls
        else:
            raise ValueError(f"Invalid types: {types}")

    @classmethod
    def get_provider(cls, alias: str) -> Optional[Type[BaseEmbeddings]]:
        """Get the embeddings class for a given provider alias."""
        return cls.provider_registry.get(alias)

    @classmethod
    def match(cls, identifier: str) -> Optional[Type[BaseEmbeddings]]:
        """Find matching embeddings class using both exact matches and patterns.

        Args:
            identifier: String to match against registry entries

        Returns:
            Matching embeddings class or None if no match found

        Examples:
            # Match exact name
            cls.match("sentence-transformers") -> SentenceTransformerEmbeddings

            # Match OpenAI pattern
            cls.match("openai://my-embedding") -> OpenAIEmbeddings

            # Match model name pattern
            cls.match("text-embedding-ada-002") -> OpenAIEmbeddings

        """
        # Firstly, we'll try to see if the provider alias is provided
        if "://" in identifier:
            provider, model_name = identifier.split("://", 1)
            if provider in cls.provider_registry:
                return cls.provider_registry[provider]

        # Now, let's try to get a match from the model name
        if identifier in cls.model_registry:
            return cls.model_registry[identifier]

        # We couldn't match the model name and there's no provider alias mentioned either. 
        # Let's try to get a match from the pattern registry
        for pattern, embeddings_cls in cls.pattern_registry.items():
            if pattern.match(identifier):
                return embeddings_cls

        # We couldn't find a proper match for the model name or the provider alias.
        return None

    @classmethod
    def wrap(cls, object: Any, **kwargs: Any) -> BaseEmbeddings:
        """Wrap an object in the appropriate embeddings class.

        The objects that are handled here could be either a Model or Client object.

        Args:
            object: Name of the embeddings implementation
            **kwargs: Additional arguments passed to the embeddings constructor

        Returns:
            Initialized embeddings instance

        """
        # Check the object type and wrap it in the appropriate embeddings class
        if isinstance(object, BaseEmbeddings):
            return object
        elif isinstance(object, str):
            embeddings_cls = cls.match(object)
            if embeddings_cls is None:
                raise ValueError(f"No matching embeddings implementation found for: {object}")
            return embeddings_cls(object, **kwargs)  # type: ignore[call-arg]
        else:
            # Loop through all the registered embeddings and check if the object is an instance of any of them
            for type_alias, embeddings_cls in cls.type_registry.items():
                if type_alias in str(type(object)):
                    return embeddings_cls(object, **kwargs)  # type: ignore[call-arg]
        raise ValueError(f"Unsupported object type for embeddings: {object}")


# Register all the available embeddings in the EmbeddingsRegistry!
# This is essential for the `AutoEmbeddings` to work properly.

# Register SentenceTransformer embeddings with pattern
EmbeddingsRegistry.register_provider("st", SentenceTransformerEmbeddings)
EmbeddingsRegistry.register_pattern(
    r"^sentence-transformers/|^all-minilm-|^paraphrase-|^multi-qa-|^msmarco-",
    SentenceTransformerEmbeddings,
)
EmbeddingsRegistry.register_types(
    "SentenceTransformer",
    SentenceTransformerEmbeddings,
)
EmbeddingsRegistry.register_model("all-minilm-l6-v2", SentenceTransformerEmbeddings)
EmbeddingsRegistry.register_model("all-mpnet-base-v2", SentenceTransformerEmbeddings)
EmbeddingsRegistry.register_model("multi-qa-mpnet-base-dot-v1", SentenceTransformerEmbeddings)
# TODO: Add all the other SentenceTranformer models here as well!

# Register OpenAI embeddings with pattern
EmbeddingsRegistry.register_provider("openai", OpenAIEmbeddings)
EmbeddingsRegistry.register_pattern(r"^text-embedding-", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-ada-002", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-3-small", OpenAIEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-3-large", OpenAIEmbeddings)

# Register model2vec embeddings
EmbeddingsRegistry.register_provider("model2vec", Model2VecEmbeddings)
EmbeddingsRegistry.register_pattern(r"^minishlab/|^minishlab/potion-base-|^minishlab/potion-|^potion-", Model2VecEmbeddings)
EmbeddingsRegistry.register_types("Model2Vec", Model2VecEmbeddings)

# Register Cohere embeddings with pattern
EmbeddingsRegistry.register_provider("cohere", CohereEmbeddings)    
EmbeddingsRegistry.register_pattern(r"^cohere|^embed-", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-multilingual-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-english-light-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register_model("embed-multilingual-v2.0", CohereEmbeddings)

# Register Jina embeddings
EmbeddingsRegistry.register_provider("jina", JinaEmbeddings)
EmbeddingsRegistry.register_pattern(r"^jina|^jinaai", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v3", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-en", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-es", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-de", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-zh", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v2-base-code", JinaEmbeddings)
EmbeddingsRegistry.register_model("jina-embeddings-v4", JinaEmbeddings)

# Register Voyage embeddings
EmbeddingsRegistry.register_provider("voyageai", VoyageAIEmbeddings)
EmbeddingsRegistry.register_pattern(r"^voyage|^voyageai", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-3-large", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-3", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-3-lite", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-code-3", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-finance-2", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-law-2", VoyageAIEmbeddings)
EmbeddingsRegistry.register_model("voyage-code-2", VoyageAIEmbeddings)

# Register Gemini embeddings
EmbeddingsRegistry.register_provider("gemini", GeminiEmbeddings)
EmbeddingsRegistry.register_pattern(r"^text-embedding-004|^embedding-001|^gemini-embedding", GeminiEmbeddings)
EmbeddingsRegistry.register_model("text-embedding-004", GeminiEmbeddings)
EmbeddingsRegistry.register_model("embedding-001", GeminiEmbeddings)
EmbeddingsRegistry.register_model("gemini-embedding-exp-03-07", GeminiEmbeddings)