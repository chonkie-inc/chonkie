"""Base class for all embeddings implementations."""

import importlib.util as importutil
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Union

# for type checking
if TYPE_CHECKING:
    import numpy as np


class BaseEmbeddings(ABC):
    """Abstract base class for all embeddings implementations.

    All embeddings implementations should inherit from this class and implement
    the embed() and similarity() methods according to their specific embedding strategy.
    """

    def __init__(self) -> None:
        """Initialize the BaseEmbeddings class.

        This class should be inherited by all embeddings classes and implement the
        abstract methods for embedding text and computing similarity between embeddings.

        It doesn't impose any specific requirements on the embedding model, because
        it can be used for embeddings providers that work via REST APIs or other means.

        Raises:
            NotImplementedError: If any of the abstract methods are not implemented

        """
        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

    @abstractmethod
    def embed(self, text: str) -> "np.ndarray":
        """Embed a text string into a vector representation.

        This method should be implemented for all embeddings models.

        Args:
            text (str): Text string to embed

        Returns:
            np.ndarray: Embedding vector for the text string

        """
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Embed a list of text strings into vector representations.

        This method should be implemented for embeddings models that support batch processing.

        By default, it calls the embed() method for each text in the list.

        Args:
            texts (List[str]): List of text strings to embed

        Returns:
            List[np.ndarray]: List of embedding vectors for each text in the list

        """
        return [self.embed(text) for text in texts]

    def _import_dependencies(self) -> None:
        """Lazy import dependencies for the embeddings implementation.

        This method should be implemented by all embeddings implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.
        """
        if self.is_available():
            global np
            import numpy as np
        else:
            raise ImportError(
                "numpy is not available. Please install it via `pip install chonkie[semantic]`"
            )

    def similarity(self, u: "np.ndarray", v: "np.ndarray") -> "np.float32":
        """Compute the similarity between two embeddings.

        Most embeddings models will use cosine similarity for this purpose. However,
        other similarity metrics can be implemented as well. Some embeddings models
        may support a similarity() method that computes the similarity between two
        embeddings via dot product or other means.

        Args:
            u (np.ndarray): First embedding vector
            v (np.ndarray): Second embedding vector

        Returns:
            np.float32: Similarity score between the two embeddings

        """
        return np.float32(np.dot(u, v.T) / (np.linalg.norm(u) * np.linalg.norm(v)))  # cosine similarity

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors.

        This property should be implemented for embeddings models that have a fixed
        dimension for their embedding vectors.

        Returns:
            int: Dimension of the embedding vectors

        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this embeddings implementation is available (dependencies installed).

        Override this method to add custom dependency checks.

        Returns:
            bool: True if the embeddings implementation is available, False otherwise

        """
        return importutil.find_spec("numpy") is not None

    @abstractmethod
    def get_tokenizer_or_token_counter(self) -> Union[Any, Callable[[str], int]]:
        """Return the tokenizer or token counter object.

        By default, this method returns the count_tokens() method, which should be
        implemented for embeddings models that require tokenization before embedding.

        Returns:
            Union[Any, Callable[[str], int]]: Tokenizer object or token counter function

        Examples:
            # Get the tokenizer object
            tokenizer = embeddings.get_tokenizer_or_token_counter()

            # Get the token counter function
            token_counter = embeddings.get_tokenizer_or_token_counter()

        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Representation of the BaseEmbeddings instance."""
        return self.__class__.__name__ + "()"

    def __call__(
        self, text: Union[str, List[str]]
    ) -> Union["np.ndarray", List["np.ndarray"]]:
        """Embed a text string into a vector representation.

        This method allows the embeddings object to be called directly with a text string
        or a list of text strings. It will call the embed() or embed_batch() method
        depending on the input type.

        Args:
            text (Union[str, List[str]]): Input text string or list of text strings

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Single or list of embedding vectors

        """
        if isinstance(text, str):
            return self.embed(text)
        elif isinstance(text, list):
            return self.embed_batch(text)
        else:
            raise ValueError("Input must be a string or list of strings")
