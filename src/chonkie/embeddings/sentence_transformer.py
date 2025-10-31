"""SentenceTransformer embeddings."""

import importlib.util as importutil
from typing import TYPE_CHECKING, Any, List, Union

import numpy as np

from .base import BaseEmbeddings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Class for SentenceTransformer embeddings.

    This class provides an interface for the SentenceTransformer library, which
    provides a variety of pre-trained models for sentence embeddings. This is also
    the recommended way to use sentence-transformers in Chonkie.

    Args:
        model (str): Name of the SentenceTransformer model to load

    """

    def __init__(
        self, 
        model: Union[str, "SentenceTransformer"] = "all-MiniLM-L6-v2",
        **kwargs: Any
    ) -> None:
        """Initialize SentenceTransformerEmbeddings with a sentence-transformers model.

        Args:
            model (str): Name of the SentenceTransformer model to load
            **kwargs: Additional keyword arguments to pass to the SentenceTransformer constructor

        Raises:
            ImportError: If sentence-transformers is not available
            ValueError: If the model is not a string or SentenceTransformer instance

        """
        super().__init__()

        # Lazy import dependencies if they are not already imported
        self._import_dependencies()

        if isinstance(model, str):
            self.model_name_or_path = model
            self.model = SentenceTransformer(self.model_name_or_path, **kwargs)
        elif isinstance(model, SentenceTransformer):
            self.model = model
            self.model_name_or_path = getattr(self.model.model_card_data, 'base_model', None) or "unknown"
        else:
            raise ValueError("model must be a string or SentenceTransformer instance")

        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text using the sentence-transformers model."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts using the sentence-transformers model."""
        return self.model.encode(texts, convert_to_numpy=True)  # type: ignore

    def embed_as_tokens(self, text: str) -> np.ndarray:
        """Embed the text as tokens using the sentence-transformers model.

        This method is useful for getting the token embeddings of a text. It
        would work even if the text is longer than the maximum sequence length.
        """
        if text == "":
            return np.array([])

        # Use the model's tokenizer to encode the text
        encoding_obj = self.model.tokenizer.encode(text, add_special_tokens=False)
        if hasattr(encoding_obj, "ids"):
            encodings = encoding_obj.ids
        else:
            encodings = encoding_obj

        max_seq_length = self.max_seq_length
        token_splits = []
        for i in range(0, len(encodings), max_seq_length):
            if i + max_seq_length <= len(encodings):
                token_splits.append(encodings[i : i + max_seq_length])
            else:
                token_splits.append(encodings[i:])

        split_texts = [self.model.tokenizer.decode(split) for split in token_splits]
        # Get the token embeddings
        try:
            token_embeddings_raw = self.model.encode(
                split_texts, output_value="token_embeddings"
            )
        except KeyError:
            # Fallback: use sentence embeddings for each split if token_embeddings not available
            token_embeddings_raw = self.model.encode(split_texts, convert_to_numpy=True)
            # Ensure all fallback embeddings are np.ndarray before expanding dims
            token_embeddings_raw = [np.expand_dims(np.array(emb), axis=0) for emb in token_embeddings_raw]  # type: ignore

        # Ensure all embeddings are numpy arrays
        token_embeddings: list[np.ndarray] = []
        if isinstance(token_embeddings_raw, list):
            for emb in token_embeddings_raw:
                if hasattr(emb, "cpu"):
                    token_embeddings.append(emb.cpu().numpy())
                else:
                    token_embeddings.append(np.array(emb))
        else:
            if hasattr(token_embeddings_raw, "cpu"):
                token_embeddings.append(token_embeddings_raw.cpu().numpy())
            else:
                token_embeddings.append(np.array(token_embeddings_raw))
        token_embeddings_np = np.concatenate(token_embeddings, axis=0)
        return token_embeddings_np

    def embed_as_tokens_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts as tokens using the sentence-transformers model."""
        return [self.embed_as_tokens(text) for text in texts]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.model.tokenizer.encode(text))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens in multiple texts using the model's tokenizer."""
        encodings = self.model.tokenizer(texts)
        return [len(enc) for enc in encodings["input_ids"]]

    def similarity(self, u: np.ndarray, v: np.ndarray) -> np.float32:
        """Compute cosine similarity between two embeddings."""
        return float(self.model.similarity(u, v).item())  # type: ignore[return-value]

    def get_tokenizer(self) -> "Tokenizer":
        """Return the tokenizer or token counter object."""
        return self.model.tokenizer

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension  # type: ignore

    @property
    def max_seq_length(self) -> int:
        """Return the maximum sequence length, using chunk_size or fallback to 512 if needed."""
        max_seq_length = self.model.get_max_seq_length()  # type: ignore
        # Try max_seq_length first
        if isinstance(max_seq_length, int):
            return max_seq_length
        # Try chunk_size if available and valid
        chunk_size = getattr(self, "chunk_size", None)
        if isinstance(chunk_size, int) and chunk_size > 0:
            return chunk_size
        # Fallback default
        return 512

    @classmethod
    def _is_available(cls) -> bool:
        """Check if sentence-transformers is available."""
        return importutil.find_spec("sentence_transformers") is not None

    @classmethod
    def _import_dependencies(cls) -> None:
        """Lazy import dependencies for the embeddings implementation.

        This method should be implemented by all embeddings implementations that require
        additional dependencies. It lazily imports the dependencies only when they are needed.
        """
        if cls._is_available():
            global SentenceTransformer
            from sentence_transformers import SentenceTransformer
        else:
            raise ImportError(
                "sentence_transformers is not available. Please install it via `pip install chonkie[st]`"
            )

    def __repr__(self) -> str:
        """Representation of the SentenceTransformerEmbeddings instance."""
        return f"SentenceTransformerEmbeddings(model={self.model_name_or_path})"
