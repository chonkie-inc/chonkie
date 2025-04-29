"""Registry for embedding implementations with pattern matching support."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Type, Union

from .base import BaseEmbeddings
from .cohere import CohereEmbeddings
from .jina import JinaEmbeddings
from .model2vec import Model2VecEmbeddings
from .openai import OpenAIEmbeddings
from .sentence_transformer import SentenceTransformerEmbeddings
from .ollama import OllamaEmbeddings


@dataclass
class RegistryEntry:
    """Registry entry containing the embeddings class and optional pattern."""

    embeddings_cls: Type[BaseEmbeddings]
    pattern: Optional[Pattern] = None
    supported_types: Optional[List[str]] = None


class EmbeddingsRegistry:
    """Registry for embedding implementations with pattern matching support."""

    _registry: Dict[str, RegistryEntry] = {}

    @classmethod
    def register(
        cls,
        name: str,
        embedding_cls: Type[BaseEmbeddings],
        pattern: Optional[Union[str, Pattern]] = None,
        supported_types: Optional[List[str]] = None,
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

        # Compile pattern if provided as string
        if isinstance(pattern, str):
            pattern = re.compile(pattern)

        cls._registry[name] = RegistryEntry(
            embeddings_cls=embedding_cls,
            pattern=pattern,
            supported_types=supported_types,
        )

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseEmbeddings]]:
        """Get embeddings class by exact name match."""
        entry = cls._registry.get(name)
        return entry.embeddings_cls if entry else None

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
        # First try exact match
        if identifier in cls._registry:
            return cls._registry[identifier].embeddings_cls

        # Then try patterns
        for entry in cls._registry.values():
            if entry.pattern and entry.pattern.match(identifier):
                return entry.embeddings_cls

        # if no match is found, raise ValueError
        raise ValueError(
            f"No matching embeddings implementation found for {identifier}"
        )

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
            return embeddings_cls(object, **kwargs)
        else:
            # Loop through all the registered embeddings and check if the object is an instance of any of them
            for entry in cls._registry.values():
                if entry.supported_types and any(
                    t in str(type(object)) for t in entry.supported_types
                ):
                    return entry.embeddings_cls(object, **kwargs)

            raise ValueError(f"Unsupported object type for embeddings: {object}")

    @classmethod
    def list_available(cls) -> List[str]:
        """List names of available embeddings implementations."""
        return [
            name
            for name, entry in cls._registry.items()
            if entry.embeddings_cls.is_available()
        ]


# Register all the available embeddings in the EmbeddingsRegistry!
# This is essential for the `AutoEmbeddings` to work properly.

# Register SentenceTransformer embeddings with pattern
EmbeddingsRegistry.register(
    "sentence-transformer",
    SentenceTransformerEmbeddings,
    pattern=r"^sentence-transformers/|^all-minilm-|^paraphrase-|^multi-qa-|^msmarco-",
    supported_types=["SentenceTransformer"],
)

# Register OpenAI embeddings with pattern
EmbeddingsRegistry.register(
    "openai", OpenAIEmbeddings, pattern=r"^openai|^text-embedding-"
)
EmbeddingsRegistry.register("text-embedding-ada-002", OpenAIEmbeddings)
EmbeddingsRegistry.register("text-embedding-3-small", OpenAIEmbeddings)
EmbeddingsRegistry.register("text-embedding-3-large", OpenAIEmbeddings)

# Register model2vec embeddings
EmbeddingsRegistry.register(
    "model2vec",
    Model2VecEmbeddings,
    pattern=r"^minishlab/|^minishlab/potion-base-|^minishlab/potion-|^potion-",
    supported_types=["Model2Vec", "model2vec"],
)

# Register Cohere embeddings with pattern
EmbeddingsRegistry.register("cohere", CohereEmbeddings, pattern=r"^cohere|^embed-")
EmbeddingsRegistry.register("embed-english-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-multilingual-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-english-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-multilingual-light-v3.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-english-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-english-light-v2.0", CohereEmbeddings)
EmbeddingsRegistry.register("embed-multilingual-v2.0", CohereEmbeddings)

# Register Ollama embeddings
EmbeddingsRegistry.register(
    "ollama",
    OllamaEmbeddings,
    pattern=r"^ollama|^snowflake-arctic-embed|^bge-|^granite-embedding"
)
EmbeddingsRegistry.register("nomic-embed-text", OllamaEmbeddings)
EmbeddingsRegistry.register("mxbai-embed-large", OllamaEmbeddings)
EmbeddingsRegistry.register("paraphrase-multilingual", OllamaEmbeddings)

# Register Jina embeddings
EmbeddingsRegistry.register("jina", JinaEmbeddings, pattern=r"^jina|^jinaai")
EmbeddingsRegistry.register("jina-embeddings-v3", JinaEmbeddings)
